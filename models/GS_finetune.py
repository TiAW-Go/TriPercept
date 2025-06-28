import torch
import numpy as np
import torch as pt
import torch.nn.parameter as nnp
from torch import nn
from torch_scatter import scatter
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import MessagePassing, TransformerConv, GlobalAttention
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, Set2Set
from torch_geometric.utils import degree

MAX_DEGREE = 4

# DeepNet: https://arxiv.org/abs/2203.00555v1
class ScaleLayer(nn.Module):
    def __init__(self, width, scale_init):
        super().__init__()
        self.scale = nnp.Parameter(pt.zeros(width) + np.log(scale_init))

    def forward(self, x):
        return pt.exp(self.scale) * x


# Graphormer: https://arxiv.org/abs/2106.05234v5
class ScaleDegreeLayer(nn.Module):
    def __init__(self, width, scale_init):
        super().__init__()
        self.scale = nnp.Parameter(pt.zeros(MAX_DEGREE, width) + np.log(scale_init))

    def forward(self, x, deg):
        return pt.exp(self.scale)[deg] * x


# GLU: https://arxiv.org/abs/1612.08083v3
class GatedLinearBlock(nn.Module):
    def __init__(self, width, num_head, scale_act, dropout=0.1, block_name=None):
        super().__init__()

        self.pre = nn.Sequential(nn.Conv1d(width, width, 1),
                                 nn.GroupNorm(num_head, width, affine=False))
        self.gate = nn.Sequential(nn.Conv1d(width, width * scale_act, 1, bias=False, groups=num_head),
                                  nn.ReLU(), nn.Dropout(dropout))
        self.value = nn.Conv1d(width, width * scale_act, 1, bias=False, groups=num_head)
        self.post = nn.Conv1d(width * scale_act, width, 1)
        if block_name is not None:
            print('##params[%s]:' % block_name, np.sum([np.prod(p.shape) for p in self.parameters()]))

    def forward(self, x):
        xx = self.pre(x.unsqueeze(-1))
        xx = self.gate(xx) * self.value(xx)
        xx = self.post(xx).squeeze(-1)
        return xx


class GatedLinearBlock2(nn.Module):
    def __init__(self, width, num_head, scale_act, dropout=0.1):
        super().__init__()

        self.gate = nn.Sequential(nn.GroupNorm(num_head, width, affine=False),
                                  nn.Conv1d(width, width * scale_act, 1, bias=False, groups=num_head),
                                  nn.ReLU(), nn.Dropout(dropout))
        self.value = nn.Sequential(nn.GroupNorm(num_head, width, affine=False),
                                   nn.Conv1d(width, width * scale_act, 1, bias=False, groups=num_head))
        self.post = nn.Conv1d(width * scale_act, width, 1)

    def forward(self, xg, xv):
        xx = self.gate(xg.unsqueeze(-1)) * self.value(xv.unsqueeze(-1))
        xx = self.post(xx).squeeze(-1)
        return xx



    # VoVNet: https://arxiv.org/abs/1904.09730v1
class ConvMessage(MessagePassing):
    def __init__(self, width, width_head, width_scale, hop, kernel, scale_init=0.1):
        super().__init__(aggr="add")
        self.width = width
        self.hop = hop

        self.bond_encoder = nn.ModuleList()
        self.pre = nn.ModuleList()
        self.msg = nn.ModuleList()
        self.scale = nn.ModuleList()
        for _ in range(hop * kernel):
            self.bond_encoder.append(BondEncoder(emb_dim=width))
            self.pre.append(nn.Linear(width, width, bias=False))
            self.msg.append(GatedLinearBlock2(width, width_head, width_scale))
            self.scale.append(ScaleDegreeLayer(width, scale_init))

        print('##params[conv]:', np.sum([np.prod(p.shape) for p in self.parameters()]))

    def forward(self, x, node_degree, edge_index, edge_attr):
        for layer in range(len(self.msg)):
            if layer == 0:
                x_raw, x_out = x, 0
            elif layer % self.hop == 0:
                x_raw, x_out = x + x_out, 0

            x_raw = self.propagate(edge_index, x=self.pre[layer](x_raw), edge_attr=edge_attr, layer=layer)
            x_out = x_out + self.scale[layer](x_raw, node_degree)
        return x_out

    def message(self, x_i, x_j, edge_attr, layer):
        bond = self.bond_encoder[layer](edge_attr)
        msg = self.msg[layer](x_i + bond, x_j + bond)
        return msg

    def update(self, aggr_out):
        return aggr_out


# GIN-virtual: https://arxiv.org/abs/2103.09430
class VirtMessage(nn.Module):
    def __init__(self, width, width_head, width_scale, scale_init=0.01):
        super().__init__()
        self.width = width

        self.msg = GatedLinearBlock(width, width_head, width_scale)
        self.scale = ScaleLayer(width, scale_init)
        print('##params[virt]:', np.sum([np.prod(p.shape) for p in self.parameters()]))

    def forward(self, x, x_res, batch, batch_size):
        xx = x_res = scatter(x, batch, dim=0, dim_size=batch_size, reduce='sum') + x_res
        xx = self.scale(self.msg(xx))[batch]
        return xx, x_res


# CosFormer: https://openreview.net/pdf?id=Bl8CQrx2Up4
class AttMessage(nn.Module):
    def __init__(self, width, width_head, width_scale, scale_init=0.01):
        super().__init__()
        self.width = width
        self.width_head = width_head

        num_grp = width // width_head
        self.pre = nn.Sequential(nn.Conv1d(width, width, 1),
                                 nn.GroupNorm(num_grp, width, affine=False))
        self.msgq = nn.Conv1d(width, width * width_scale, 1, bias=False, groups=num_grp)
        self.msgk = nn.Conv1d(width, width * width_scale, 1, bias=False, groups=num_grp)
        self.msgv = nn.Conv1d(width, width * width_scale, 1, bias=False, groups=num_grp)
        self.post = nn.Conv1d(width * width_scale, width, 1)
        self.scale = ScaleLayer(width, scale_init)
        print('##params[att]:', np.sum([np.prod(p.shape) for p in self.parameters()]))

    def forward(self, x, x_res, batch, batch_size):
        xv = self.pre(x.unsqueeze(-1))

        shape = [len(x), -1, self.width_head]
        xq = pt.exp(self.msgq(xv) / np.sqrt(self.width_head)).reshape(shape)
        xk = pt.exp(self.msgk(xv) / np.sqrt(self.width_head)).reshape(shape)
        xv = self.msgv(xv).reshape(shape)

        xv = pt.einsum('bnh,bnv->bnhv', xk, xv)
        xv = x_res = scatter(xv, batch, dim=0, dim_size=batch_size, reduce='sum') + x_res
        xk = scatter(xk, batch, dim=0, dim_size=batch_size, reduce='sum')[batch]
        xq = xq / pt.einsum('bnh,bnh->bn', xq, xk)[:, :, None]  # norm
        xv = pt.einsum('bnh,bnhv->bnv', xq, xv[batch]).reshape(len(x), -1, 1)

        xv = self.scale(self.post(xv).squeeze(-1))
        return xv, x_res

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


class Encoder(nn.Module):
    def __init__(self, num_layers, emb_dim, conv_hop, conv_kernel):
        super().__init__()
        self.num_layers = num_layers
        self.atom_encoder = AtomEncoder(emb_dim)
        # self.bond_encoder = GaussianLayer(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)
        self.scale = ScaleDegreeLayer(emb_dim, 0.1)

        self.conv = pt.nn.ModuleList()
        self.virt = pt.nn.ModuleList()
        self.att = pt.nn.ModuleList()
        self.main = pt.nn.ModuleList()
        for layer in range(num_layers):
            self.conv.append(ConvMessage(emb_dim, 16, 1, conv_hop, conv_kernel))
            self.virt.append(VirtMessage(emb_dim, 16, 2))
            self.att.append(AttMessage(emb_dim, 16, 2))
            self.main.append(GatedLinearBlock(emb_dim, 16, 3))  # debug

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch, edge_distance = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch, batched_data.distance_bin
        batch_size = len(batched_data.ptr) - 1
        node_degree = degree(edge_index[1], len(x)).long() - 1
        node_degree.clamp_(0, MAX_DEGREE - 1)
        h_in = self.bond_encoder(edge_attr)
        h_in = scatter(h_in, edge_index[1], dim=0, dim_size=len(x), reduce='sum')
        h_in = self.scale(h_in, node_degree)
        h_in, h_att, h_virt = self.atom_encoder(x) + h_in, 0, 0

        for layer in range(self.num_layers):
            h_out = h_in + self.conv[layer](h_in, node_degree, edge_index, edge_attr)
            #if virt[layer]
            h_tmp, h_virt = self.virt[layer](h_in, h_virt, batch, batch_size)
            h_out, h_tmp = h_out + h_tmp, None
            # att[layer]
            h_tmp, h_att = self.att[layer](h_in, h_att, batch, batch_size)
            h_out, h_tmp = h_out + h_tmp, None

            h_out = h_in = self.main[layer](h_out)
        return h_out


class GINet(pt.nn.Module):

    def __init__(self, task = "classification", num_layers = 5, emb_dim=300, feat_dim=256, conv_hop = 3, conv_kernel = 2, graph_pooling = "sum", num_tasks = 1):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''
        super(GINet, self).__init__()

        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.task = task

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        self.gnn_node = Encoder(num_layers, emb_dim, conv_hop, conv_kernel)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = pt.nn.Sequential(pt.nn.Linear(emb_dim, emb_dim), pt.nn.BatchNorm1d(emb_dim), pt.nn.ReLU(), pt.nn.Linear(emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        self.norm = pt.nn.GroupNorm(1, self.emb_dim, affine=False)
        if graph_pooling == "set2set":
            self.head = pt.nn.Linear(self.emb_dim*2, self.num_tasks)
        else:
            self.head = pt.nn.Linear(self.emb_dim, self.num_tasks)

        # 全连接映射层，将图级表示映射到预测特征维度
        self.feat_lin = nn.Linear(emb_dim, feat_dim)

        # 预测头（根据任务不同选择分类或回归）
        if self.task == 'classification':
            self.pred_head = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2),
                nn.Softplus(),
                nn.Linear(feat_dim // 2, num_tasks*2)
            )
        elif self.task == 'regression':
            self.pred_head = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2),
                nn.Softplus(),
                nn.Linear(feat_dim // 2, 1)
            )
        else:
            raise ValueError("Task not recognized: choose 'classification' or 'regression'.")

    def forward(self, batched_data):
        h = self.gnn_node(batched_data)
        h_pool = self.pool(h, batched_data.batch)
        h_graph = self.feat_lin(h_pool)
        h_out = self.pred_head(h_graph)
        # h_graph = self.norm(h_pool)
        # h_out = self.head(h_graph)
        # if not self.training: h_out.clamp_(0, 20)  #是否保留

        return h_graph, h_out

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                param = param.data
            own_state[name].copy_(param)
