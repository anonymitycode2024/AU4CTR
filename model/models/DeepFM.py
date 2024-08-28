from model.BasiclLayer import BasicCTR, BasicCL4CTR, FactorizationMachine, MultiLayerPerceptron, FeaturesLinear
from model.AU4CTR import StarCL
import torch.nn as nn

class DeepFM_CL(StarCL):
    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), dropout=0.5,pratio=0.2):
        super(DeepFM_CL, self).__init__(field_dims, embed_dim, pratio)
        self.lr = FeaturesLinear(field_dims=field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embed_output_size = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_size, mlp_layers, dropout, output_layer=False)
        self.fc = nn.Linear(mlp_layers[-1], 1)

        self.mlp_ = MultiLayerPerceptron(self.embed_output_size, mlp_layers, dropout, output_layer=False)

    def encoder_copy(self, x_emb):
        fi_out = self.mlp_(x_emb.view(x_emb.size(0), -1))
        return fi_out
        
    def feature_interaction(self, x_emb):
        fi_out = self.mlp(x_emb.view(x_emb.size(0), -1))
        return fi_out
    
    def forward(self, x):
        """
        :param x: B,F
        :return:
        """
        x_emb = self.embedding(x) 
        fi_out = self.feature_interaction(x_emb)
        x_out = self.lr(x) + self.fm(x_emb) + self.fc(fi_out)
        return x_out

class DeepFM(BasicCTR):
    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), dropout=0.5):
        super(DeepFM, self).__init__(field_dims, embed_dim)
        self.lr = FeaturesLinear(field_dims=field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embed_output_size = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_size, mlp_layers, dropout, output_layer=True)

    def forward(self, x):
        """
        :param x: B,F
        :return:
        """
        x_embed = self.embedding(x)  # B,F,E
        x_out = self.lr(x) + self.fm(x_embed) + self.mlp(x_embed.view(x.size(0), -1))
        return x_out


class DeepFM_CL4CTR(BasicCL4CTR):
    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), dropout=0.5, batch_size=1024, pratio=0.5,
                 fi_type="att"):
        super(DeepFM_CL4CTR, self).__init__(field_dims, embed_dim, batch_size, pratio=pratio, fi_type=fi_type)
        self.lr = FeaturesLinear(field_dims=field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embed_output_size = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_size, mlp_layers, dropout, output_layer=True)

    def forward(self, x):
        x_embed = self.embedding(x)  # B,F,E
        x_out = self.lr(x) + self.fm(x_embed) + self.mlp(x_embed.view(x.size(0), -1))
        return x_out