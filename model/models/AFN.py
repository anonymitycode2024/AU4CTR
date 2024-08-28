import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.BasiclLayer import FeaturesEmbedding, MultiLayerPerceptron, FeaturesLinear
from model.AU4CTR import StarCL


class LNN(torch.nn.Module):
    def __init__(self, num_fields, embed_dim, LNN_dim, bias=False):
        super(LNN, self).__init__()
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.LNN_dim = LNN_dim
        self.lnn_output_dim = LNN_dim * embed_dim

        self.weight = torch.nn.Parameter(torch.Tensor(LNN_dim, num_fields))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(LNN_dim, embed_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields, embedding_size)``
        """
        embed_x_abs = torch.abs(x)  # Computes the element-wise absolute value of the given input tensor.
        embed_x_afn = torch.add(embed_x_abs, 1e-7)
        # Logarithmic Transformation
        embed_x_log = torch.log1p(embed_x_afn)  # torch.log1p
        lnn_out = torch.matmul(self.weight, embed_x_log)
        if self.bias is not None:
            lnn_out += self.bias

        # and torch.expm1
        lnn_exp = torch.expm1(lnn_out)
        output = F.relu(lnn_exp).contiguous().view(-1, self.lnn_output_dim)
        return output

class AFN_CL(StarCL):
    def __init__(self, field_dims, embed_dim, LNN_dim=10, num_layers=3, dropouts=(0.5, 0.5),pratio=0.5):
        super().__init__(field_dims, embed_dim, pratio)
        self.linear = FeaturesLinear(field_dims)
        mlp_dims = [400]*num_layers
        self.num_fields = len(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)  # Embedding

        self.LNN_dim = LNN_dim
        self.LNN_output_dim = self.LNN_dim * embed_dim
        self.LNN = LNN(self.num_fields, embed_dim,  LNN_dim)

        self.LNN_c = LNN(self.num_fields, embed_dim,  LNN_dim)


        self.mlp = MultiLayerPerceptron(self.LNN_output_dim, mlp_dims, dropouts[0])

    def encoder_copy(self,x_emb):
        lnn_out = self.LNN_c(x_emb)
        return lnn_out
    
    def feature_interaction(self,x_emb):
        lnn_out = self.LNN(x_emb)
        return lnn_out

    def forward(self, x):
        x_emb = self.embedding(x)

        fi_out = self.feature_interaction(x_emb)

        pred_y = self.mlp(fi_out) + self.linear(x)
        return pred_y    

class AFNPlus_CL(StarCL):
    def __init__(self, field_dims, embed_dim, LNN_dim=100, mlp_dims=(400, 400, 400),
                 mlp_dims2=(400, 400, 400), dropouts=(0.5, 0.5), pratio=0.5):
        super().__init__(field_dims, embed_dim, pratio)
        self.num_fields = len(field_dims)
        # self.linear = FeaturesLinear(field_dims)  # Linear
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)  # Embedding

        self.LNN_dim = LNN_dim
        self.LNN_output_dim = self.LNN_dim * embed_dim
        self.LNN = LNN(self.num_fields, embed_dim, LNN_dim)
        self.LNN_c = LNN(self.num_fields, embed_dim, LNN_dim)

        self.mlp = MultiLayerPerceptron(self.LNN_output_dim, mlp_dims, dropouts[0], output_layer=False)
        self.mlp_c = MultiLayerPerceptron(self.LNN_output_dim, mlp_dims, dropouts[0], output_layer=False)

        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp2 = MultiLayerPerceptron(self.embed_output_dim, mlp_dims2, dropouts[1], output_layer=False)
        self.mlp2_c = MultiLayerPerceptron(self.embed_output_dim, mlp_dims2, dropouts[1], output_layer=False)
        
        self.fc = nn.Linear(mlp_dims[-1]+mlp_dims2[-1], 1)
    
    def encoder_copy(self, x_emb):
        lnn_out = self.LNN_c(x_emb)
        x_lnn = self.mlp_c(lnn_out)
        x_dnn = self.mlp2_c(x_emb.view(-1, self.embed_output_dim))
        fi_out = torch.cat([x_lnn, x_dnn], dim=1)
        return fi_out

    
    def feature_interaction(self,x_emb):
        lnn_out = self.LNN(x_emb)
        x_lnn = self.mlp(lnn_out)
        x_dnn = self.mlp2(x_emb.view(-1, self.embed_output_dim))
        fi_out = torch.cat([x_lnn, x_dnn], dim=1)
        return fi_out


    def forward(self, x):

        x_emb = self.embedding(x)

        fi_out = self.feature_interaction(x_emb)
        pred_y = self.fc(fi_out)

        return pred_y


class AFNPlus(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, LNN_dim=100, mlp_dims=(400, 400, 400),
                 mlp_dims2=(400, 400, 400), dropouts=(0.5, 0.5)):
        super().__init__()
        self.num_fields = len(field_dims)
        # self.linear = FeaturesLinear(field_dims)  # Linear
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)  # Embedding

        self.LNN_dim = LNN_dim
        self.LNN_output_dim = self.LNN_dim * embed_dim
        self.LNN = LNN(self.num_fields, embed_dim, LNN_dim)

        self.mlp = MultiLayerPerceptron(self.LNN_output_dim, mlp_dims, dropouts[0])

        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp2 = MultiLayerPerceptron(self.embed_output_dim, mlp_dims2, dropouts[1], output_layer=True)


    def forward(self, x):

        x_emb = self.embedding(x)

        lnn_out = self.LNN(x_emb)
        x_lnn = self.mlp(lnn_out)

        x_dnn = self.mlp2(x_emb.view(-1, self.embed_output_dim))

        pred_y = x_dnn + x_lnn

        return pred_y
    

class AFN(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, LNN_dim=10, num_layers=3, dropouts=(0.5, 0.5)):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        mlp_dims = [400]*num_layers
        self.num_fields = len(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)  # Embedding

        self.LNN_dim = LNN_dim
        self.LNN_output_dim = self.LNN_dim * embed_dim
        self.LNN = LNN(self.num_fields, embed_dim,  LNN_dim)

        self.mlp = MultiLayerPerceptron(self.LNN_output_dim, mlp_dims, dropouts[0])

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x_emb = self.embedding(x)

        lnn_out = self.LNN(x_emb)

        pred_y = self.mlp(lnn_out) + self.linear(x)
        return pred_y