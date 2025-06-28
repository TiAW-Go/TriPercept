import math
import random
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import gc


ATOM_LIST = list(range(1,119))


class MoleculeDataset(Dataset):
    def __init__(self, data_path):
        super(Dataset, self).__init__()
        self.graph_data, self.slices = torch.load(data_path)

    def __getitem__(self, index):
        # 获取原子特征的切片起始和结束位置
        start_x = self.slices['x'][index]
        end_x = self.slices['x'][index + 1]

        # 获取分子标签的切片起始和结束位置
        start_y = self.slices['y'][index]
        end_y = self.slices['y'][index + 1]

        # 获取边的切片起始和结束位置
        start_edge = self.slices['edge_index'][index]
        end_edge = self.slices['edge_index'][index + 1]

        # 根据切片信息加载节点特征、边索引、边特征和标签
        x = self.graph_data.x[start_x:end_x]  # 节点特征（九维）
        edge_index = self.graph_data.edge_index[:, start_edge:end_edge]  # 边索引
        edge_attr = self.graph_data.edge_attr[start_edge:end_edge]  # 边特征（三维）
        edge_distance = self.graph_data.distance_bin[start_edge:end_edge]  # 边的距离
        # print("edge_distance:",edge_distance)
        y = self.graph_data.y[start_y:end_y]  # 目标值

        N = x.size(0)  # 节点数
        M = int(edge_index.size(1) / 2)  # 边数

        # 随机掩码部分
        num_mask_nodes = max([1, math.floor(0.25 * N)])  # 随机掩码节点的数量
        num_mask_edges = max([0, math.floor(0.25 * M)])  # 随机掩码边的数量

        # 随机选择掩码节点和边
        mask_nodes_i = random.sample(list(range(N)), num_mask_nodes)
        mask_nodes_j = random.sample(list(range(N)), num_mask_nodes)
        mask_edges_i_single = random.sample(list(range(M)), num_mask_edges)
        mask_edges_j_single = random.sample(list(range(M)), num_mask_edges)
        mask_edges_i = [2 * i for i in mask_edges_i_single] + [2 * i + 1 for i in mask_edges_i_single]
        mask_edges_j = [2 * i for i in mask_edges_j_single] + [2 * i + 1 for i in mask_edges_j_single]

        # 更新掩码后的节点特征
        x_i = x.clone()  # 使用 clone() 避免深拷贝
        for atom_idx in mask_nodes_i:
            x_i[atom_idx, :] = torch.tensor([len(ATOM_LIST)]+ [0]*8)  # 替换掩码节点特征（九维）

        # 更新掩码后的边索引和边特征
        edge_index_i = torch.zeros((2, 2 * (M - num_mask_edges)), dtype=torch.long)
        edge_attr_i = torch.zeros((2 * (M - num_mask_edges), 3), dtype=torch.long)  # 三维边特征
        edge_distance_i = torch.zeros(2 * (M - num_mask_edges), dtype=torch.float)

        count = 0
        for bond_idx in range(2 * M):
            if bond_idx not in mask_edges_i:
                edge_index_i[:, count] = edge_index[:, bond_idx]
                edge_attr_i[count, :] = edge_attr[bond_idx, :]
                edge_distance_i[count] = edge_distance[bond_idx]
                count += 1
        # print("edge_distance_i:",edge_distance_i)
        data_i = Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i, edge_distance=edge_distance_i, y=y)

        # 同样更新掩码后的节点特征和边索引
        x_j = x.clone()  # 使用 clone() 避免深拷贝
        for atom_idx in mask_nodes_j:
            x_j[atom_idx, :] = torch.tensor([len(ATOM_LIST)]+ [0]*8)  # 替换掩码节点特征（九维）

        edge_index_j = torch.zeros((2, 2 * (M - num_mask_edges)), dtype=torch.long)
        edge_attr_j = torch.zeros((2 * (M - num_mask_edges), 3), dtype=torch.long)  # 三维边特征
        edge_distance_j = torch.zeros(2 * (M - num_mask_edges), dtype=torch.float)

        count = 0
        for bond_idx in range(2 * M):
            if bond_idx not in mask_edges_j:
                edge_index_j[:, count] = edge_index[:, bond_idx]
                edge_attr_j[count, :] = edge_attr[bond_idx, :]
                edge_distance_j[count] = edge_distance[bond_idx]
                count += 1

        data_j = Data(x=x_j, edge_index=edge_index_j, edge_attr=edge_attr_j, edge_distance=edge_distance_j, y=y)

        # 强制进行垃圾回收以释放内存
        # gc.collect()

        return data_i, data_j

    def __len__(self):
        return len(self.slices['x']) - 1  # 返回图的数量

    def get(self, index):
        pass

    def len(self):
        pass


class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, data_path):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size

    def get_data_loaders(self):
        train_dataset = MoleculeDataset(data_path=self.data_path)

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        print("train_loader:", len(train_loader))
        print("valid_loader:", len(valid_loader))
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        # num_train = len(train_dataset)
        num_train = 3108187
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        return train_loader, valid_loader
