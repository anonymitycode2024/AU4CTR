import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.BasiclLayer import FeaturesEmbedding,FactorizationMachine, MultiLayerPerceptron

class DIFM(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_layers=(256,256,256), num_heads=8):
        super(DIFM, self).__init__()
        self.lin = FeaturesLinearWeight(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        self.dual_fen = DualFENLayer(len(field_dims), embed_dim, embed_dims=mlp_layers, num_heads=num_heads)

        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb, x_weight = self.dual_fen(x_emb)
        x_lin = self.lin(x, x_weight.unsqueeze(2))
        pred_y = self.fm(x_emb) + x_lin
        return pred_y


class FeaturesLinearWeight(torch.nn.Module):
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array(
            (0, *np.cumsum(field_dims)[:-1]), dtype=np.float64)

    def forward(self, x, weight=None):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)

        return torch.sum(torch.mul(self.fc(x), weight),dim=1) + self.bias
    

class DualFENLayer(nn.Module):
    def __init__(self, field_length, embed_dim, embed_dims=(256, 256, 256), att_size=64, num_heads=8):
        super(DualFENLayer, self).__init__()

        input_dim = field_length * embed_dim  
        self.mlp = MultiLayerPerceptron(input_dim, embed_dims, dropout=0.5, output_layer=False)
        self.multihead = MultiHeadAttentionL(model_dim=embed_dim, dk=att_size, num_heads=num_heads)
        self.trans_vec_size = att_size * num_heads * field_length
        self.trans_vec = nn.Linear(self.trans_vec_size, field_length, bias=False)
        self.trans_bit = nn.Linear(embed_dims[-1], field_length, bias=False)

    def forward(self, x_emb):
        x_con = x_emb.view(x_emb.size(0), -1)  # [B, ?]

        m_bit = self.mlp(x_con)  # [B,F]

        x_att2 = self.multihead(x_emb, x_emb, x_emb)  # B,dk*n*f
        m_vec = self.trans_vec(x_att2.view(-1, self.trans_vec_size))  # B,F
        m_bit = self.trans_bit(m_bit)

        x_att = m_bit + m_vec  # B,F
        x_emb = x_emb * x_att.unsqueeze(2)
        return x_emb, x_att
    

class MultiHeadAttentionL(nn.Module):
    def __init__(self, model_dim=256, dk=32, num_heads=16):
        super(MultiHeadAttentionL, self).__init__()

        self.dim_per_head = dk  
        self.num_heads = num_heads

        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.linear_residual = nn.Linear(model_dim, self.dim_per_head * num_heads)


    def _dot_product_attention(self, q, k, v, scale=None):
        attention = torch.bmm(q, k.transpose(1, 2)) * scale
        ## score = softmax(QK^T / (d_k ** 0.5))
        attention = torch.softmax(attention, dim=2)
        attention = torch.dropout(attention, p=0.0, train=self.training)
        context = torch.bmm(attention, v)
        return context, attention

    def forward(self, key0, value0, query0, attn_mask=None):
        batch_size = key0.size(0)

        key = self.linear_k(key0)  # K = UWk [B, 10, 256*16]
        value = self.linear_v(value0)  # Q = UWv [B, 10, 256*16]
        query = self.linear_q(query0)  # V = UWq [B, 10, 256*16]

        key = key.view(batch_size * self.num_heads, -1, self.dim_per_head)
        value = value.view(batch_size * self.num_heads, -1, self.dim_per_head)
        query = query.view(batch_size * self.num_heads, -1, self.dim_per_head)

        scale = (key.size(-1) // self.num_heads) ** -0.5
        context, attention = self._dot_product_attention(query, key, value, scale)
        context = context.view(batch_size, -1, self.dim_per_head * self.num_heads)  # [B, 10, 256*h]

        residual = self.linear_residual(query0)
        residual = residual.view(batch_size, -1, self.dim_per_head * self.num_heads)  # [B, 10, 256*h]

        output = torch.relu(residual + context)  # [B, 10, 256] difm synax 7
        return output
