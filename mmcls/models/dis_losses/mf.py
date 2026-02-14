import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from mmcls.registry import MODELS

# --- Helper Function: Distributed Sinkhorn ---
@torch.no_grad()
def distributed_sinkhorn(out, nmb_iters=3, epsilon=0.05, world_size=1):
    Q = torch.exp(out / epsilon).permute(0, 2, 1)  # K-by-B
    B = Q.shape[2] * world_size 
    K = Q.shape[1] 

    # make the matrix sums to 1
    sum_Q = Q.sum(dim=(1, 2), keepdim=True)
    if dist.is_initialized():
        dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(nmb_iters):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=2, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

    Q *= B 
    return Q.permute(0, 2, 1)

# --- KoLeo Loss Classes (Dependencies) ---
class KoLeoLossData(nn.Module):
    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        dots = torch.bmm(x, x.transpose(1, 2))
        dots.diagonal(dim1=-2, dim2=-1).fill_(-1)
        _, I = torch.max(dots, dim=2)
        return I

    def forward(self, student_output, eps=1e-8):
        # student_output: (B, T, feat_dim) or (1, K, D) in your specific usage
        with torch.cuda.amp.autocast(enabled=False):
            I = self.pairwise_NNs_inner(student_output)
            B, T, feat_dim = student_output.shape
            batch_indices = torch.arange(B, device=student_output.device).view(-1, 1).expand(-1, T)
            neighbors = student_output[batch_indices, I]
            
            flat_student = student_output.view(-1, feat_dim)
            flat_neighbors = neighbors.view(-1, feat_dim)
            distances = self.pdist(flat_student, flat_neighbors)
            loss = -torch.log(distances + eps).mean()
        return loss

class KoLeoLossPrototypes(nn.Module):
    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)
        _, I = torch.max(dots, dim=1)
        return I

    def forward(self, student_output, eps=1e-8):
        with torch.cuda.amp.autocast(enabled=False):
            I = self.pairwise_NNs_inner(student_output)
            distances = self.pdist(student_output, student_output[I])
            loss = -torch.log(distances + eps).mean()
        return loss

# --- Main Loss Class ---
@MODELS.register_module()
class MFLoss(nn.Module):
    """
    Manifold Loss with Prototypes & KoLeo Regularization.
    Applied to the First (0), Second (1), and Last (-1) layers using identical dimensions.
    """
    def __init__(self,
                 name,
                 use_this,
                 student_dims,           # int: Dimension of student features (same for all layers)
                 teacher_dims,           # int: Dimension of teacher features (same for all layers)
                 K=6304,                 # Number of prototypes
                 temperature=0.1,
                 sinkhorn_iters=3,
                 normalize_input=True,
                 weight_mf=1.0,          # Weight for Main Manifold Loss
                 weight_koleo_data=0.0,  # Weight for KoLeo Data Loss
                 weight_koleo_proto=0.0, # Weight for KoLeo Proto Loss
                 world_size=4):
        super(MFLoss, self).__init__()

        self.K = K
        self.temp = temperature
        self.sinkhorn_iters = sinkhorn_iters
        self.normalize_input = normalize_input
        self.world_size = world_size
        
        # Loss Weights
        self.weight_mf = weight_mf
        self.weight_koleo_data = weight_koleo_data
        self.weight_koleo_proto = weight_koleo_proto

        # Specifically targeting: First, Second, and Last layers
        self.target_indices = 3
        
        # Initialize 3 separate Projectors and 3 separate sets of Prototypes
        # Since dims are the same, we just repeat the logic 3 times.
        self.projectors = nn.ModuleList()
        self.prototypes = nn.ParameterList()

        for _ in range(3):
            # Projector: Student Dim -> Teacher Dim
            self.projectors.append(nn.Linear(student_dims, teacher_dims, bias=False))
            
            # Prototypes: (K, Teacher Dim)
            proto = torch.empty(K, teacher_dims)
            _sqrt_k = (1. / teacher_dims) ** 0.5
            torch.nn.init.uniform_(proto, -_sqrt_k, _sqrt_k)
            self.prototypes.append(nn.Parameter(proto))

        # Sub-loss modules
        self.koleo_data_loss = KoLeoLossData()
        self.koleo_proto_loss = KoLeoLossPrototypes()

    def forward(self, preds_S, preds_T):

    # ================== DEBUGGING BLOCK START ==================
    # Only print on the main process (Rank 0) to avoid messy logs
    rank = 0
    if dist.is_initialized():
        rank = dist.get_rank()
        
    if rank == 0:
        print(f"\n[DEBUG] MFLoss Parameter Check:")
        # Initialize storage for previous weights if it doesn't exist
        if not hasattr(self, 'debug_weights'):
            self.debug_weights = {}

        # Iterate over the 3 layers
        for i in range(len(self.projectors)):
            # Get current weights (Data only, no gradients)
            curr_proj = self.projectors[i].weight.data
            curr_proto = self.prototypes[i].data

            # 1. Print Mean and Std
            print(f"  Layer {i} | Projector : Mean {curr_proj.mean():.6f} | Std {curr_proj.std():.6f}")
            print(f"  Layer {i} | Prototypes: Mean {curr_proto.mean():.6f} | Std {curr_proto.std():.6f}")

            # 2. Check for updates (compare with previous step)
            if f'proj_{i}' in self.debug_weights:
                prev_proj = self.debug_weights[f'proj_{i}']
                prev_proto = self.debug_weights[f'proto_{i}']

                # Calculate absolute difference
                diff_proj = (curr_proj - prev_proj).abs().sum().item()
                diff_proto = (curr_proto - prev_proto).abs().sum().item()

                # Print status
                # Note: Small diffs (e.g., 0.0001) mean it IS updating. 0.0 means it is FROZEN.
                status_proj = "UPDATED" if diff_proj > 1e-9 else "FROZEN (Check Optimizer!)"
                status_proto = "UPDATED" if diff_proto > 1e-9 else "FROZEN (Check Optimizer!)"

                print(f"    -> Proj Diff: {diff_proj:.8f} [{status_proj}]")
                print(f"    -> Proto Diff: {diff_proto:.8f} [{status_proto}]")

            # 3. Store current weights for the next forward pass
            self.debug_weights[f'proj_{i}'] = curr_proj.clone()
            self.debug_weights[f'proto_{i}'] = curr_proto.clone()
        print("========================================================\n")
        # ================== DEBUGGING BLOCK END ==================
        """
        Args:
            preds_S (List[Tensor]): Student features list.
            preds_T (List[Tensor]): Teacher features list.
        """
        total_loss_mf = 0.0
        total_loss_koleo_d = 0.0
        total_loss_koleo_p = 0.0

        s_low = preds_S[0]
        s_high = preds_S[1]
        
        t_low = preds_T[0]
        t_high = preds_T[1]

        feats_S = [s_low[:, 0], s_low[:, 1], s_high]
        feats_T = [t_low[:, 0], t_low[:, 1], t_high]

        # Loop 3 times for indices 0, 1, and -1
        for i in len(self.target_indices):
            F_s = feats_S[i]
            F_t = feats_T[i]
            
            # 1. Normalize Prototypes (In-place)
            with torch.no_grad():
                self.prototypes[i].copy_(F.normalize(self.prototypes[i], dim=1))

            # 2. Sampling (Random Permutation)
            bsz, patch_num, _ = F_s.shape
            total_patches = bsz * patch_num
            # Safety check: if we have fewer patches than K, use all available patches
            num_samples = min(self.K, total_patches) 
            
            sampler = torch.randperm(total_patches, device=F_s.device)[:num_samples]

            f_s = F_s.reshape(total_patches, -1)[sampler].unsqueeze(0) # (1, K, D_s)
            f_t = F_t.reshape(total_patches, -1)[sampler].unsqueeze(0) # (1, K, D_t)

            # 3. Input Normalization (Optional)
            if self.normalize_input:
                eps = 1e-8
                f_s = (f_s - f_s.mean(dim=1, keepdim=True)) / (f_s.std(dim=1, keepdim=True) + eps)
                f_t = (f_t - f_t.mean(dim=1, keepdim=True)) / (f_t.std(dim=1, keepdim=True) + eps)

            # 4. Projection
            f_s = self.projectors[i](f_s)

            # L2 Normalize
            f_s = F.normalize(f_s, dim=-1, p=2)
            f_t = F.normalize(f_t, dim=-1, p=2)

            # 5. KoLeo Losses
            l_koleo_d = self.koleo_data_loss(f_s)
            l_koleo_p = self.koleo_proto_loss(self.prototypes[i])

            # 6. Manifold (Sinkhorn) Loss
            # Student vs Prototypes
            M_s = f_s @ self.prototypes[i].t() # (1, K, K)
            q1 = distributed_sinkhorn(M_s, nmb_iters=self.sinkhorn_iters, world_size=self.world_size).detach()
            
            # Teacher vs Prototypes
            M_t = f_t @ self.prototypes[i].t()
            q2 = distributed_sinkhorn(M_t, nmb_iters=self.sinkhorn_iters, world_size=self.world_size).detach()

            # Softmax
            p1 = F.softmax(M_s / self.temp, dim=2)
            p2 = F.softmax(M_t / self.temp, dim=2)

            # Consistency Loss
            loss12 = - torch.mean(torch.sum(q1 * torch.log(p2 + 1e-6), dim=2))
            loss21 = - torch.mean(torch.sum(q2 * torch.log(p1 + 1e-6), dim=2))
            
            l_mf = (loss12 + loss21) / 2

            # Aggregate
            total_loss_mf += l_mf
            total_loss_koleo_d += l_koleo_d
            total_loss_koleo_p += l_koleo_p

        # Final weighted sum
        final_loss = (self.weight_mf * total_loss_mf + 
                      self.weight_koleo_data * total_loss_koleo_d + 
                      self.weight_koleo_proto * total_loss_koleo_p) / self.target_indices

        return final_loss
