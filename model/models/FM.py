import torch
import torch.nn as nn
from model.BasiclLayer import BasicCTR, BasicCL4CTR, FactorizationMachine, FeaturesLinear
from model.AU4CTR import StarCL

class FM(BasicCTR):
    def __init__(self, field_dims, embed_dim):
        super(FM, self).__init__(field_dims, embed_dim)
        self.lr = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        emb_x = self.embedding(x)
        x = self.lr(x) + self.fm(emb_x)
        return x


class FM_CL4CTR(BasicCL4CTR):
    def __init__(self, field_dims, embed_dim, batch_size=4096, pratio=0.5, fi_type="att"):
        super(FM_CL4CTR, self).__init__(field_dims, embed_dim, batch_size, pratio=pratio, fi_type=fi_type)
        self.lr = FeaturesLinear(field_dims=field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        emb_x = self.embedding(x)
        x = self.lr(x) + self.fm(emb_x)
        return x

class FM_CL(StarCL):
    def __init__(self, field_dims, embed_dim, pratio=0.5):
        super(FM_CL, self).__init__(field_dims, embed_dim, pratio)
        self.lr = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=False)
        self.projection = nn.Linear(embed_dim, embed_dim)
            
    
    def encoder_copy(self,x_emb):
        fm_inter = self.fm(x_emb)
        return fm_inter
    
    def feature_interaction(self, x_emb):
        fm_inter = self.fm(x_emb)
        return fm_inter

    def forward(self, x):
        x_emb = self.embedding(x)
        fi_out = self.feature_interaction(x_emb)
        pred_y = self.lr(x) + torch.sum(fi_out, dim=1, keepdim=True)
        return pred_y
