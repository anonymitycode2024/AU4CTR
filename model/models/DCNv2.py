import torch
import torch.nn as nn

from model.BasiclLayer import FeaturesEmbedding, MultiLayerPerceptron, BasicCL4CTR
from model.AU4CTR import StarCL

class CNV2_CL(StarCL):
    def __init__(self, field_dims, embed_dim, cn_layers=3, pratio=0.5):
        super(CNV2_CL, self).__init__(field_dims, embed_dim, pratio)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        if isinstance(embed_dim, int):
            self.embed_output_dim = len(field_dims) * embed_dim
        else:
            self.embed_output_dim = sum(embed_dim)
        self.cross_net = CrossNetworkV2(self.embed_output_dim, cn_layers)

        self.fc = torch.nn.Linear(self.embed_output_dim, 1)
    
    def feature_interaction(self, x_emb):
        x_emb = x_emb.view(-1, self.embed_output_dim)
        cross_cn = self.cross_net(x_emb)
        return cross_cn


    def forward(self, x):
        x_emb = self.embedding(x)
        fi_out = self.feature_interaction(x_emb)
        pred_y = self.fc(fi_out)
        return pred_y
    
class DCNV2P_CL(StarCL):
    def __init__(self, field_dims, embed_dim, cn_layers=3, mlp_layers=(400, 400, 400), dropout=0.5, pratio=0.5):
        super(DCNV2P_CL, self).__init__(field_dims, embed_dim, pratio)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # self.embed_output_dim = len(field_dims) * embed_dim
        self.embed_output_dim = len(field_dims) * embed_dim
        
        self.cross_net = CrossNetworkV2(self.embed_output_dim, cn_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_layers, output_layer=False, dropout=dropout)
        self.fc = torch.nn.Linear(mlp_layers[-1] + self.embed_output_dim, 1)
        
        self.cross_net_ = CrossNetworkV2(self.embed_output_dim, cn_layers)
        self.mlp_ = MultiLayerPerceptron(self.embed_output_dim, mlp_layers, output_layer=False, dropout=dropout)
        

    def encoder_copy(self, x_emb): 
        x_emb = x_emb.view(-1, self.embed_output_dim)
        cross_cn = self.cross_net_(x_emb)
        cross_mlp = self.mlp_(x_emb)
        return torch.cat([cross_cn, cross_mlp], dim=1)
    
    def feature_interaction(self, x_emb):
        x_emb = x_emb.view(-1, self.embed_output_dim)
        cross_cn = self.cross_net(x_emb)
        cross_mlp = self.mlp(x_emb)
        return torch.cat([cross_cn, cross_mlp], dim=1)

    def forward(self, x):
        x_emb = self.embedding(x) 
        fi_out = self.feature_interaction(x_emb)
        pred_y = self.fc(fi_out)
        return pred_y

class DCNV2P_CL4CTR(BasicCL4CTR):
    def __init__(self, field_dims, embed_dim, cn_layers=3, mlp_layers=(400, 400, 400), dropout=0.5,batch_size=4096, pratio=0.5, fi_type="att"):
        super(DCNV2P_CL4CTR, self).__init__(field_dims, embed_dim, batch_size, pratio=pratio, fi_type=fi_type)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cross_net = CrossNetworkV2(self.embed_output_dim, cn_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_layers, output_layer=False, dropout=dropout)
        self.fc = torch.nn.Linear(mlp_layers[-1] + self.embed_output_dim, 1)

    def forward(self, x):
        x_emb = self.embedding(x).view(-1, self.embed_output_dim)
        cross_cn = self.cross_net(x_emb)
        cross_mlp = self.mlp(x_emb)

        pred_y = self.fc(torch.cat([cross_cn, cross_mlp], dim=1))
        return pred_y

class DCNV2P(nn.Module):
    def __init__(self, field_dims, embed_dim, cn_layers=3, mlp_layers=(400, 400, 400), dropout=0.5):
        super(DCNV2P, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cross_net = CrossNetworkV2(self.embed_output_dim, cn_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_layers, output_layer=False, dropout=dropout)
        self.fc = torch.nn.Linear(mlp_layers[-1] + self.embed_output_dim, 1)

    def forward(self, x):
        x_emb = self.embedding(x).view(-1, self.embed_output_dim)  # B,F*E
        cross_cn = self.cross_net(x_emb)
        cross_mlp = self.mlp(x_emb)

        pred_y = self.fc(torch.cat([cross_cn, cross_mlp], dim=1))
        return pred_y

class CrossNetworkV2(nn.Module):
    def __init__(self, input_dim, cn_layers):
        super().__init__()

        self.cn_layers = cn_layers

        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, input_dim, bias=False) for _ in range(cn_layers)
        ])
        self.b = torch.nn.ParameterList([torch.nn.Parameter(
            torch.zeros((input_dim,))) for _ in range(cn_layers)])

    def forward(self, x):
        x0 = x
        for i in range(self.cn_layers):
            xw = self.w[i](x)
            x = x0 * (xw + self.b[i]) + x
        return x