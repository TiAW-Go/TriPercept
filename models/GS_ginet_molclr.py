import torch
import torch as pt
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import degree, add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_scatter import scatter
import torch.nn.parameter as nnp
from torch_geometric.nn import MessagePassing
import copy


MAX_DEGREE = 4


num_atom_type = 119
num_chirality_tag = 3
num_degree = 11
num_formal_charge = 9
num_hybridization = 7
num_aromatic = 4
num_hydrogen = 6
num_implicit_valence = 2
num_in_ring = 2

num_bond_type = 5
num_bond_stereo = 3
num_conjugated = 2

class AtomEncoderAttention(nn.Module):
    def __init__(self, emb_dim):
        super(AtomEncoderAttention, self).__init__()
        self.embedding0 = nn.Embedding(num_atom_type, emb_dim)
        self.embedding1 = nn.Embedding(num_chirality_tag, emb_dim)
        self.embedding2 = nn.Embedding(num_degree, emb_dim)
        self.embedding3 = nn.Embedding(num_formal_charge, emb_dim)
        self.embedding4 = nn.Embedding(num_hybridization, emb_dim)
        self.embedding5 = nn.Embedding(num_aromatic, emb_dim)
        self.embedding6 = nn.Embedding(num_hydrogen, emb_dim)
        self.embedding7 = nn.Embedding(num_implicit_valence, emb_dim)
        self.embedding8 = nn.Embedding(num_in_ring, emb_dim)

        nn.init.xavier_uniform_(self.embedding0.weight)
        nn.init.xavier_uniform_(self.embedding1.weight)
        nn.init.xavier_uniform_(self.embedding2.weight)
        nn.init.xavier_uniform_(self.embedding3.weight)
        nn.init.xavier_uniform_(self.embedding4.weight)
        nn.init.xavier_uniform_(self.embedding5.weight)
        nn.init.xavier_uniform_(self.embedding6.weight)
        nn.init.xavier_uniform_(self.embedding7.weight)
        nn.init.xavier_uniform_(self.embedding8.weight)

        self.att_vector = nn.Parameter(torch.randn(emb_dim))
        self.out_linear = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        """
        :param x: LongTensor, shape = [num_atoms, 9]
                  每列依次为：原子序数, 手性标签, 度, 形式电荷, 杂化状态, 芳香性, 氢原子数, 隐式价态, 是否在环中
        :return: Tensor, shape = [num_atoms, emb_dim]，即每个原子的表示
        """

        emb0 = self.embedding0(x[:, 0])
        emb1 = self.embedding1(x[:, 1])
        emb2 = self.embedding2(x[:, 2])
        emb3 = self.embedding3(x[:, 3])
        emb4 = self.embedding4(x[:, 4])
        emb5 = self.embedding5(x[:, 5])
        emb6 = self.embedding6(x[:, 6])
        emb7 = self.embedding7(x[:, 7])
        emb8 = self.embedding8(x[:, 8])

        embeddings = torch.stack([emb0, emb1, emb2, emb3, emb4, emb5, emb6, emb7, emb8], dim=1)
        scores = (embeddings * self.att_vector.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        att_weights = torch.softmax(scores, dim=1)
        fused = (embeddings * att_weights.unsqueeze(-1)).sum(dim=1)
        out = self.out_linear(fused)
        return out

class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_stereo, emb_dim)
        self.edge_embedding3 = nn.Embedding(num_conjugated, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding3.weight.data)

    def forward(self, x, edge_index, edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]
        self_loop_attr = torch.zeros(x.size(0), 3, device=edge_attr.device, dtype=edge_attr.dtype)
        self_loop_attr[:, 0] = 4
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + \
                          self.edge_embedding2(edge_attr[:, 1]) + \
                          self.edge_embedding3(edge_attr[:, 2])
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class ScaleDegreeLayer(nn.Module):
    def __init__(self, width, scale_init):
        super().__init__()
        self.scale = nnp.Parameter(pt.zeros(MAX_DEGREE, width) + np.log(scale_init))

    def forward(self, x, deg):
        return pt.exp(self.scale)[deg] * x

@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(nn.Module):
    def __init__(self, K=128):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        nn.init.uniform_(self.means.weight, 2, 3)
        nn.init.uniform_(self.stds.weight, 1, 1.5)

    def forward(self, x):
        x = x.unsqueeze(-1)
        mean = self.means.weight.float().view(-1).unsqueeze(0)
        std = (self.stds.weight.float().view(-1).abs() + 1e-2).unsqueeze(0)
        x = gaussian(x.float(), mean, std).type_as(self.means.weight)
        return x

class GINet(nn.Module):
    def __init__(self, num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, pool='mean'):
        super(GINet, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio

        self.bond_encoder = GaussianLayer(emb_dim)
        self.x_embedding = AtomEncoderAttention(emb_dim)
        self.scale = ScaleDegreeLayer(emb_dim, 0.1)

        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool

        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)
        self.out_lin = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim // 2)
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        distance_bin = data.edge_distance

        node_degree = degree(edge_index[1], len(x)).long() - 1
        node_degree.clamp_(0, MAX_DEGREE - 1)
        h_in = self.bond_encoder(distance_bin)
        h_in = scatter(h_in, edge_index[1], dim=0, dim_size=len(x), reduce='sum')
        h_in = self.scale(h_in, node_degree)
        h = self.x_embedding(x) + h_in

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        out = self.out_lin(h)
        return h, out

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                param = param.data
            own_state[name].copy_(param)



class EMA():

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


class MERIT(nn.Module):
    def __init__(self, args, moving_average_decay):
        super().__init__()

        self.online_encoder = GINet(emb_dim=args["emb_dim"], feat_dim=args["feat_dim"])
        self.target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(self.target_encoder, False)
        self.target_ema_updater = EMA(moving_average_decay)

    def update_ma(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self):
        model2 = self.target_encoder
        return model2