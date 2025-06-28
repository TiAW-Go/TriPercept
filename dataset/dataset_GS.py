import os
import csv
import math
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch_scatter import scatter
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger                                                                                                                                                               

RDLogger.DisableLog('rdApp.*')

ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.smiles_data):
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, seed=11, log_every_n=1000):
    print("seed:",seed)
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    if seed is not None:
        random.seed(seed)
        random.shuffle(scaffold_sets)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)

    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set

    return train_inds, valid_inds, test_inds




def read_smiles(data_path, target, task):
    smiles_data, labels = [], []
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            smiles = row['smiles']
            label = row[target]
            mol = Chem.MolFromSmiles(smiles)
            if mol != None:
                if task == 'classification':
                    if label:
                        smiles_data.append(smiles)
                        labels.append(int(label))
                    else:
                        labels.append(-1)
                elif task == 'regression':
                    if label:
                        smiles_data.append(smiles)
                        labels.append(float(label))
                    else:
                        labels.append(-1)
                else:
                    ValueError('task must be either regression or classification')
    return smiles_data, labels


class MolTestDataset(Dataset):
    def __init__(self, data_path, target, task):
        super(Dataset, self).__init__()
        pt_path = (data_path+'.pt')
        csv_path = (data_path+'.csv')
        print("pt_path:",pt_path)
        print("csv_path:",csv_path)
        self.task = task
        self.conversion = 1
        if 'qm9' in csv_path and target in ['homo', 'lumo', 'gap', 'zpve', 'u0']:
            self.conversion = 27.211386246
            print(target, 'Unit conversion needed!')

        self.data, self.slices = torch.load(pt_path)
        self.smiles_data, self.labels = read_smiles(csv_path, target, task)
        self.valid_indices = [i for i, lab in enumerate(self.labels) if lab != -1]

    def __getitem__(self, index):
        if self.task == 'classification':
            true_index = self.valid_indices[index]
        elif self.task == 'regression':
            true_index = index
        data = self.data.__class__()
        for key in self.data.keys:
            s = self.slices[key][true_index].item()
            e = self.slices[key][true_index + 1].item()
            if key == 'edge_index':
                data[key] = self.data[key][:, s:e]
            else:
                data[key] = self.data[key][s:e]
        if self.task == 'classification':
            data.y = torch.tensor(self.labels[true_index], dtype=torch.long).view(-1,1)
        elif self.task == 'regression':
            data.y = torch.tensor(self.labels[true_index]*self.conversion, dtype=torch.float).view(-1,1)
        return data

    def __len__(self):
        return len(self.smiles_data)

    def get(self, index):
        pass

    def len(self):
        pass



class MolTestDatasetWrapper(object):
    
    def __init__(self, 
        batch_size, num_workers, valid_size, test_size, 
        data_path, target, task, splitting, seed
    ):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.target = target
        self.task = task
        self.splitting = splitting
        self.seed = seed
        assert splitting in ['random', 'scaffold']

    def get_data_loaders(self):
        train_dataset = MolTestDataset(data_path=self.data_path, target=self.target, task=self.task)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, train_dataset):
        if self.splitting == 'random':
            num_train = len(train_dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)

            split = int(np.floor(self.valid_size * num_train))
            split2 = int(np.floor(self.test_size * num_train))
            valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]
        
        elif self.splitting == 'scaffold':
            train_idx, valid_idx, test_idx = scaffold_split(train_dataset, self.valid_size, self.test_size, self.seed)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        valid_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        test_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=test_sampler,
            num_workers=self.num_workers, drop_last=False
        )

        return train_loader, valid_loader, test_loader
