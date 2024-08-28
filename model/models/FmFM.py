import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.BasiclLayer import FactorizationMachine, FeaturesEmbedding
from model.AU4CTR import StarCL

class FMFM(nn.Module):
    def __init__(self, field_dims, embed_dim, interaction_type="matrix"):
        super(FMFM, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.num_field = len(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.inter_num = self.num_field * (self.num_field - 1) // 2
        self.field_interaction_type = interaction_type
        if self.field_interaction_type == "vector":  # FvFM
            self.interaction_weight = nn.Parameter(torch.Tensor(self.inter_num, embed_dim))
        elif self.field_interaction_type == "matrix":  # FmFM
            self.interaction_weight = nn.Parameter(torch.Tensor(self.inter_num, embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.interaction_weight.data)
        self.row, self.col = list(), list()
        for i in range(self.num_field - 1):
            for j in range(i + 1, self.num_field):
                self.row.append(i), self.col.append(j)

    def forward(self, x):

        x_emb = self.embedding(x)
        left_emb = x_emb[:, self.row]
        right_emb = x_emb[:, self.col]
        if self.field_interaction_type == "vector":
            left_emb = left_emb * self.interaction_weight  # B,I,E
        elif self.field_interaction_type == "matrix":
            # B,I,1,E * I,E,E = B,I,1,E => B,I,E
            left_emb = torch.matmul(left_emb.unsqueeze(2), self.interaction_weight).squeeze(2)
        pred_y = (left_emb * right_emb).sum(dim=-1).sum(dim=-1, keepdim=True)
        return pred_y
    

class FMFM_CL(StarCL):
    def __init__(self, field_dims, embed_dim, interaction_type="matrix", pratio=0.2):
        super(FMFM_CL, self).__init__(field_dims, embed_dim, pratio)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.num_field = len(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.inter_num = self.num_field * (self.num_field - 1) // 2
        self.field_interaction_type = interaction_type
        if self.field_interaction_type == "vector":  # FvFM
            self.interaction_weight = nn.Parameter(torch.Tensor(self.inter_num, embed_dim))
            self.interaction_weight_ = nn.Parameter(torch.Tensor(self.inter_num, embed_dim))

        elif self.field_interaction_type == "matrix":  # FmFM
            self.interaction_weight = nn.Parameter(torch.Tensor(self.inter_num, embed_dim, embed_dim))
            self.interaction_weight_ = nn.Parameter(torch.Tensor(self.inter_num, embed_dim, embed_dim))

        nn.init.xavier_uniform_(self.interaction_weight.data)
        self.row, self.col = list(), list()
        for i in range(self.num_field - 1):
            for j in range(i + 1, self.num_field):
                self.row.append(i), self.col.append(j)
       
    def encoder_copy(self, x_emb):
        left_emb = x_emb[:, self.row]
        right_emb = x_emb[:, self.col]
        if self.field_interaction_type == "vector":
            left_emb = left_emb * self.interaction_weight_  # B,I,E
        elif self.field_interaction_type == "matrix":
            left_emb = torch.matmul(left_emb.unsqueeze(2), self.interaction_weight_).squeeze(2)
        inter = (left_emb * right_emb).sum(dim=-1) #B,I
        return inter     

    def feature_interaction(self, x_emb):
        left_emb = x_emb[:, self.row]
        right_emb = x_emb[:, self.col]
        if self.field_interaction_type == "vector":
            left_emb = left_emb * self.interaction_weight  # B,I,E
        elif self.field_interaction_type == "matrix":
            left_emb = torch.matmul(left_emb.unsqueeze(2), self.interaction_weight).squeeze(2)
        inter = (left_emb * right_emb).sum(dim=-1) #B,I
        return inter

    def forward(self, x):
        x_emb = self.embedding(x)
        fi_out = self.feature_interaction(x_emb)
        pred_y = fi_out.sum(dim=-1, keepdim=True)
        return pred_y

