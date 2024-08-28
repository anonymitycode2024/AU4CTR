import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
from .BasiclLayer import FeaturesEmbedding

class StarCL(nn.Module): 
    def __init__(self,field_dims, embed_dim, batch_size=1024, temp=1, pratio=0.5):
        super(StarCL,self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_dim = embed_dim
        self.field_dims = field_dims
        self.sum_field = sum(field_dims)
        self.pratio = pratio
        self.dp = nn.Dropout(p=pratio)
        self.field_dims_cum = np.array((0, *np.cumsum(self.field_dims)))        
    
    def forward(self):
        raise NotImplementedError
    
    def feature_interaction(self, x_emb): 
        raise NotImplementedError

    def encoder_copy(self, x_emb): 
        raise NotImplementedError

    def compute_cl_loss_all(self, x, lambda_au=0.001, beta=0.001, lambda_i=0.001): # 
        L_align,L_uni = self.compute_alignment_loss() 
        if lambda_i == 0.0:
            return L_align * lambda_au  + L_uni * beta
        L_CL_loss = self.compute_self_loss(x) 
        L_cl = L_align * lambda_au  + L_uni * beta + L_CL_loss * lambda_i
        return L_cl
    
    def compute_cl_loss_two(self, lambda_au=0.001, beta=0.001):
        L_align,L_uni = self.compute_alignment_loss() 
        L_cl = L_align * lambda_au  + L_uni * beta
        return L_cl

    def compute_cl_loss_self(self, x, lambda_i=0.001):
        return lambda_i * self.compute_self_loss(x) 

    def compute_self_loss(self, x): 
        x_emb = self.embedding(x) 
        dp_emb = self.dp(x_emb) 
        
        fi_vector = self.feature_interaction(x_emb)
        fi_vector_dp = self.encoder_copy(dp_emb) 
         # fi_vector = fi_vector.detach() # 
        fi_vector.detach_()  
        cl_loss = torch.norm(fi_vector_dp.sub(fi_vector), dim=1).pow(2).mean() # (3)Self-alignment loss
        return cl_loss 

    def compute_alignment_loss(self,a=2):
        L_align = 0.0
        embedds = self.embedding.embedding.weight
        V_K_Es= []
        for start, end in zip(self.field_dims_cum[:-1], self.field_dims_cum[1:]):
            embed_f = embedds[int(start):int(end), :]  
            embed_f = F.normalize(embed_f, p=2, dim=1)
            V_K_E = embed_f.mean(dim=0) # 1,E # V_K_E = F.normalize(V_K_E1, p=2, dim=0)
            V_K_Es.append(V_K_E) # 
            L_align += torch.norm(embed_f.sub(V_K_E),p=2).pow(a).sum()
        L_align = L_align/self.sum_field
        V_K_Es = torch.stack(V_K_Es,0)
        L_uni = self.compute_uniformity_loss(V_K_Es)
        return L_align, L_uni
        

    def compute_uniformity_loss(self, V_K_Es, t = 1):
        sq_dist = torch.pdist(V_K_Es, p=2).pow(2) 
        return sq_dist.mul(-t).exp().mean().log()