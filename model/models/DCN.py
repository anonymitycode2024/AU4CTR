import torch
import torch.nn as nn
from model.BasiclLayer import FeaturesEmbedding, MultiLayerPerceptron
from model.AU4CTR import StarCL


class CN_CL(StarCL):
    def __init__(self, field_dims, embed_dim, cn_layers=3, pratio=0.5):
        super(CN_CL, self).__init__(field_dims, embed_dim, pratio) 
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cross_network = CrossNetwork(self.embed_output_dim, cn_layers) # self.fi 
        self.fc = nn.Linear(self.embed_output_dim, 1)
    
        self.cross_network_copy = CrossNetwork(self.embed_output_dim, cn_layers) # self.fi 
    
    def encoder_copy(self, x_emb): 
        x_emb = x_emb.reshape(-1, self.embed_output_dim)
        cross_cn = self.cross_network_copy(x_emb)    
        return cross_cn
    
    def feature_interaction(self,x_emb):
        x_emb = x_emb.reshape(-1, self.embed_output_dim)
        cross_cn = self.cross_network(x_emb)    
        return cross_cn

    def forward(self, x):
        x_emb = self.embedding(x)
        pred_y = self.fc(self.feature_interaction(x_emb) )
        return pred_y
    

class DCN_CL(StarCL):
    def __init__(self, field_dims, embed_dim, cn_layers=3, mlp_layers=(400, 400, 400), dropout=0.5, pratio=0.5):
        super(DCN_CL, self).__init__(field_dims,embed_dim, pratio)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cross_network = CrossNetwork(self.embed_output_dim, cn_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_layers, output_layer=False, dropout=dropout)
        
        self.fc = nn.Linear(mlp_layers[-1] + self.embed_output_dim, 1)
    
        self.cross_network_c = CrossNetwork(self.embed_output_dim, cn_layers)
        self.mlp_c = MultiLayerPerceptron(self.embed_output_dim, mlp_layers, output_layer=False, dropout=dropout)

    def encoder_copy(self,x_emb):
        x_emb = x_emb.reshape(-1, self.embed_output_dim)
        cross_cn = self.cross_network_c(x_emb)
        cross_mlp = self.mlp_c(x_emb)        
        fi_vector = torch.cat([cross_cn, cross_mlp], dim=1)
        return fi_vector  

    def feature_interaction(self,x_emb):
        x_emb = x_emb.reshape(-1, self.embed_output_dim)
        cross_cn = self.cross_network(x_emb)
        cross_mlp = self.mlp(x_emb)        
        fi_vector = torch.cat([cross_cn, cross_mlp], dim=1)
        return fi_vector
        
    def forward(self, x):
        x_emb = self.embedding(x)
        fi_out = self.feature_interaction(x_emb)
        pred_y = self.fc(fi_out)
        return pred_y
    


class CrossNetwork(nn.Module):
    def __init__(self, input_dim, cn_layers):
        super().__init__()

        self.cn_layers = cn_layers

        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(cn_layers)
        ])
        self.b = torch.nn.ParameterList([torch.nn.Parameter(
            torch.zeros((input_dim,))) for _ in range(cn_layers)])

    def forward(self, x):
        x0 = x
        for i in range(self.cn_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x

class DCN(nn.Module):
    def __init__(self, field_dims, embed_dim, cn_layers=3, mlp_layers=(400, 400, 400), dropout=0.5):
        super(DCN, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim

        self.fi = CrossNetwork(self.embed_output_dim, cn_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_layers, output_layer=False, dropout=dropout)
        self.fc = nn.Linear(mlp_layers[-1] + self.embed_output_dim, 1)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb = x_emb.reshape(-1, self.embed_output_dim)
        cross_cn = self.fi(x_emb)
        cross_mlp = self.mlp(x_emb)
        pred_y = self.fc(torch.cat([cross_cn, cross_mlp], dim=1))
        return pred_y

class CN(nn.Module):
    def __init__(self, field_dims, embed_dim, cn_layers=3):
        super(CN, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.fi = CrossNetwork(self.embed_output_dim, cn_layers)
        self.fc = nn.Linear(self.embed_output_dim, 1)

    def forward(self, x):
        x_emb = self.embedding(x).view(-1, self.embed_output_dim)
        cross_cn = self.fi(x_emb)
        pred_y = self.fc(cross_cn)
        return pred_y