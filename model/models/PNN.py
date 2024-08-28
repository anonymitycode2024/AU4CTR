import torch
import torch.nn as nn

from model.BasiclLayer  import FeaturesEmbedding, MultiLayerPerceptron, BasicCL4CTR
from model.AU4CTR import StarCL


class InnerProductNetwork(nn.Module):
    def __init__(self, num_fields):
        super(InnerProductNetwork, self).__init__()
        self.row, self.col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                self.row.append(i), self.col.append(j)

    def forward(self, x):
        return torch.sum(x[:, self.row] * x[:, self.col], dim=2)

class IPNN_CL(StarCL):
    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), dropout=0.5, pratio=0.5):
        super(IPNN_CL, self).__init__(field_dims, embed_dim, pratio)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        num_fields = len(field_dims)
        self.pnn = InnerProductNetwork(num_fields)

        self.embed_output_dim = num_fields * embed_dim
        self.inter_size = num_fields * (num_fields - 1) // 2
        self.fc = MultiLayerPerceptron(self.inter_size + self.embed_output_dim, mlp_layers, dropout=dropout)
        
        self.projection = nn.Linear(self.inter_size, embed_dim)
    
        self.pnn_ = InnerProductNetwork(num_fields)
    
    def encoder_copy(self, x_emb):
        cross_ipnn = self.pnn_(x_emb)  # B,1/2* nf*(nf-1)
        return cross_ipnn

    def feature_interaction(self, x_emb):
        cross_ipnn = self.pnn(x_emb)  # B,1/2* nf*(nf-1)
        return cross_ipnn
    
    def forward(self, x):
        x_emb = self.embedding(x)  # B,n_f,embed
        fi_out = self.feature_interaction(x_emb)
        x = torch.cat([x_emb.view(-1, self.embed_output_dim), fi_out], dim=1)
        pred_y = self.fc(x) 
        return pred_y
    

class IPNN_CL4CTR(BasicCL4CTR):
    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), dropout=0.5, pratio=0.3):
        super(IPNN_CL4CTR, self).__init__(field_dims, embed_dim, batch_size=4096, pratio=pratio)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        num_fields = len(field_dims)
        self.pnn = InnerProductNetwork(num_fields)

        self.embed_output_dim = num_fields * embed_dim
        self.inter_size = num_fields * (num_fields - 1) // 2
        self.fc = MultiLayerPerceptron(self.inter_size + self.embed_output_dim, mlp_layers, dropout=dropout)

    def feature_interaction(self, x_emb):
        cross_ipnn = self.pnn(x_emb)  # B,1/2* nf*(nf-1)
        return cross_ipnn
    
    def forward(self, x):
        x_emb = self.embedding(x)  # B,n_f,embed
        fi_out = self.feature_interaction(x_emb)
        x = torch.cat([x_emb.view(-1, self.embed_output_dim), fi_out], dim=1)
        pred_y = self.fc(x) 
        return pred_y
    

class IPNN(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), dropout=0.5):
        super(IPNN, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        num_fields = len(field_dims)
        self.pnn = InnerProductNetwork(num_fields)

        self.embed_output_dim = num_fields * embed_dim
        self.inter_size = num_fields * (num_fields - 1) // 2
        self.fc = MultiLayerPerceptron(self.inter_size + self.embed_output_dim, mlp_layers, dropout=dropout)

    def feature_interaction(self, x_emb):
        cross_ipnn = self.pnn(x_emb)  # B,1/2* nf*(nf-1)
        return cross_ipnn
    
    def forward(self, x):
        x_emb = self.embedding(x)  # B,n_f,embed
        fi_out = self.feature_interaction(x_emb)
        x = torch.cat([x_emb.view(-1, self.embed_output_dim), fi_out], dim=1)
        pred_y = self.fc(x) 
        return pred_y

class OPNN(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), dropout=0.5, kernel_type="vec"):
        super(OPNN, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        num_fields = len(field_dims)
        if kernel_type == "original":
            # 没有参数的
            self.pnn = OuterProductNetwork2(num_fields, embed_dim)
        else:
            self.pnn = OuterProductNetwork(num_fields, embed_dim, kernel_type)

        self.embed_output_dim = num_fields * embed_dim
        self.inter_size = num_fields * (num_fields - 1) // 2
        self.fc = MultiLayerPerceptron(self.inter_size + self.embed_output_dim, mlp_layers, dropout)
        
    def feature_interaction(self, x_emb):
        cross_opnn = self.pnn(x_emb)  # B,1/2* nf*(nf-1)
        return cross_opnn
    
    def forward(self, x):
        x_emb = self.embedding(x)  # B,n_f,embed
        fi_out = self.feature_interaction(x_emb)
        x = torch.cat([x_emb.view(-1, self.embed_output_dim), fi_out], dim=1)
        pred_y = self.fc(x) 
        return pred_y


class OPNN_CL(StarCL):
    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), dropout=0.5, kernel_type="vec",pratio=0.5):
        super(OPNN_CL, self).__init__(field_dims, embed_dim, pratio)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        num_fields = len(field_dims)
        if kernel_type == "original":
            # 没有参数的
            self.pnn = OuterProductNetwork2(num_fields, embed_dim)
        else:
            self.pnn = OuterProductNetwork(num_fields, embed_dim, kernel_type)

        self.embed_output_dim = num_fields * embed_dim
        self.inter_size = num_fields * (num_fields - 1) // 2
        self.fc = MultiLayerPerceptron(self.inter_size + self.embed_output_dim, mlp_layers, dropout)
        
    def feature_interaction(self, x_emb):
        cross_opnn = self.pnn(x_emb)  # B,1/2* nf*(nf-1)
        return cross_opnn
    
    def forward(self, x):
        x_emb = self.embedding(x)  # B,n_f,embed
        fi_out = self.feature_interaction(x_emb)
        x = torch.cat([x_emb.view(-1, self.embed_output_dim), fi_out], dim=1)
        pred_y = self.fc(x) 
        return pred_y
    

class OuterProductNetwork(nn.Module):
    def __init__(self, num_fields, embed_dim, kernel_type='mat'):
        super().__init__()
        num_ix = num_fields * (num_fields - 1) // 2
        if kernel_type == 'mat':
            kernel_shape = embed_dim, num_ix, embed_dim  # E,F*(F-1)/2,E. 每一对特征都有一个矩阵，相当于fibinet的each
        elif kernel_type == 'vec':
            kernel_shape = num_ix, embed_dim  # all
        elif kernel_type == 'num':
            kernel_shape = num_ix, 1  # all 单个权重
        else:
            raise ValueError('unknown kernel type: ' + kernel_type)
        self.kernel_type = kernel_type
        self.kernel = torch.nn.Parameter(torch.zeros(kernel_shape))
        torch.nn.init.xavier_uniform_(self.kernel.data)

        self.row, self.col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                self.row.append(i), self.col.append(j)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        p, q = x[:, self.row], x[:, self.col]
        if self.kernel_type == 'mat':
            #  p [b,1,num_ix,e]
            #  kernel [e, num_ix, e]
            kp = torch.sum(p.unsqueeze(1) * self.kernel, dim=-1).permute(0, 2, 1)  # b,num_ix,e
            return torch.sum(kp * q, -1)
        else:
            # p * q [B,ix,E] * [1,ix,E] => B,ix,E
            return torch.sum(p * q * self.kernel.unsqueeze(0), -1)


class OuterProductNetwork2(nn.Module):
    def __init__(self, num_fields, embed_dim, kernel_type='num'):
        super().__init__()
        self.row, self.col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                self.row.append(i), self.col.append(j)

    def forward(self, x):
        p, q = x[:, self.row], x[:, self.col]
        # B,IX,E,1     B,IX,1,E
        p, q = p.unsqueeze(-1), q.unsqueeze(2)
        pq = torch.matmul(p, q)  # B,IX,E,E
        pq = torch.sum(torch.sum(pq, dim=-1), dim=-1)  # B,IX
        return pq
