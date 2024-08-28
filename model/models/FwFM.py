import torch
import torch.nn as nn
from model.BasiclLayer import FeaturesEmbedding,FeaturesLinear
from model.AU4CTR import StarCL

class FwFM(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super(FwFM, self).__init__()
        self.lr = FeaturesLinear(field_dims)
        # self.lr_w = FeaturesLinearWeight(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.fwfm = FwFMInterLayer_(len(field_dims))

    def forward(self, x):
        x_emb = self.embedding(x)  
        pred_y = self.lr(x) + self.fwfm(x_emb)
        return pred_y


class FwFM_CL(StarCL):
    def __init__(self, field_dims, embed_dim, pratio=0.5):
        super(FwFM_CL, self).__init__(field_dims, embed_dim, pratio=0.5)
        self.lr = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        num_fields = len(field_dims)
        num_inter = (num_fields * (num_fields - 1)) // 2

        self.fc = nn.Linear(num_inter, 1)
        self.fwfm = FwFMInterLayer(len(field_dims))

        self.projection = nn.Linear(num_inter, embed_dim)

        
    
    def feature_interaction(self, x_emb):
        x_inter = self.fwfm(x_emb)
        return x_inter

    def forward(self, x):
        x_emb = self.embedding(x)  
        pred_y = self.lr(x) + self.fc(self.feature_interaction(x_emb))
        return pred_y

class FwFMInterLayer_(nn.Module):
    def __init__(self, num_fields):
        super(FwFMInterLayer_, self).__init__()

        self.num_fields = num_fields
        num_inter = (num_fields * (num_fields - 1)) // 2

        self.fc = nn.Linear(num_inter, 1)
        self.row, self.col = list(), list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                self.row.append(i), self.col.append(j)

    def forward(self, x_embed):
        x_inter = torch.sum(x_embed[:, self.row] * x_embed[:, self.col], dim=2, keepdim=False)  # [B,n*(n-1)/2]
        inter_sum = self.fc(x_inter)
        return inter_sum


class FwFMInterLayer(nn.Module):
    def __init__(self, num_fields):
        super(FwFMInterLayer, self).__init__()
        self.num_fields = num_fields
        self.row, self.col = list(), list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                self.row.append(i), self.col.append(j)

    def forward(self, x_embed):
        x_inter = torch.sum(x_embed[:, self.row] * x_embed[:, self.col], dim=2, keepdim=False)  # [B,n*(n-1)/2]
        return x_inter