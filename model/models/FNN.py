import torch
import torch.nn as nn
from model.BasiclLayer  import FeaturesEmbedding, MultiLayerPerceptron
from model.AU4CTR import StarCL


class FNN(nn.Module):
    def __init__(self, field_dims, embed_dim, num_layers=3, mlp_layers=(400, 400, 400), dropout=0.5):
        super(FNN, self).__init__()
        mlp_layers = [400] * num_layers
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, embed_dims=mlp_layers,
                                        dropout=dropout, output_layer=True)


    def forward(self, x):
        x_emb = self.embedding(x)
        pred_y = self.mlp(x_emb.view(x.size(0), -1))
        return pred_y

class FNN_CL(StarCL):
    def __init__(self, field_dims, embed_dim, num_layers=3, mlp_layers=(400, 400, 400), dropout=0.5, pratio=0.3):
        super(FNN_CL, self).__init__(field_dims, embed_dim, pratio)
        mlp_layers = [400] * num_layers
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, embed_dims=mlp_layers,
                                        dropout=dropout, output_layer=False)
        self.fc = nn.Linear(mlp_layers[-1], 1)
        
        self.mlp_ = MultiLayerPerceptron(self.embed_output_dim, embed_dims=mlp_layers,
                                dropout=dropout, output_layer=False)

    def encoder_copy(self, x_emb):
        fi_out = self.mlp_(x_emb.view(x_emb.size(0), -1))
        return fi_out
    
    def feature_interaction(self, x_emb):
        fi_out = self.mlp(x_emb.view(x_emb.size(0), -1))
        return fi_out

    def forward(self, x):
        x_emb = self.embedding(x)
        fi_out = self.feature_interaction(x_emb)
        pred_y = self.fc(fi_out)
        return pred_y