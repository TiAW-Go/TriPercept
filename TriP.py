import os
import shutil
import sys
import torch
import yaml
import numpy as np
from datetime import datetime

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.avge import AverageMeter
from utils.nt_xent import NTXentLoss
from tqdm import tqdm
# from apex import amp
apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class DIG_Mol(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        dir_name = datetime.now().strftime('%b%d_%H-%M-%S')

        log_dir = os.path.join('ckpt', dir_name)

        self.writer = SummaryWriter(log_dir=log_dir)

        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])


    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']

            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, xis, xjs, n_iter):
        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        # 特征向量标准化
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def _step_n2(self, model, xis, xjs, n_iter):
        # get the representations and the projections
        with torch.no_grad():
            ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        with torch.no_grad():
            rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        # 特征向量标准化
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        # self.nt_xent_criterion不作为函数 为何可以进行计算
        return loss

    def _step_n1n2(self, model_1, model_2, xis, xjs, n_iter):
        # get the representations and the projections
        ris, zis = model_1(xis)  # [N,C]
        # print("ris.shape:",ris.shape,"zis.shape:",zis.shape)  #ris.shape: torch.Size([256, 256]) zis.shape: torch.Size([256, 128])
        # get the representations and the projections
        with torch.no_grad():


            rjs, zjs = model_2(xjs)  # [N,C]
        # print("rjs.shape:",rjs.shape,"zjs.shape:",zjs.shape)  #rjs.shape: torch.Size([256, 256]) zjs.shape: torch.Size([256, 128])
        # normalize projection feature vectors
        # 特征向量标准化
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        # print("zis1.shape:",zis.shape,"zjs1.shape:",zjs.shape)  #zis1.shape: torch.Size([256, 128]) zjs1.shape: torch.Size([256, 128])
        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def train(self):
        train_loader, valid_loader = self.dataset.get_data_loaders()
        # print(self.dataset.get_data_loaders.__code__.co_filename)  #/home/aita3660/AIDD/hw/DIG-Mol-main/dataset/dataset.py
        # print("train_loader:1",train_loader)
        # for batch_idx, batch in enumerate(train_loader):
        #     print(f"Batch {batch_idx}:")
        #     print(batch)
        #     #修改前：[DataBatch(x=[6109, 2], edge_index=[2, 9992], edge_attr=[9992, 2], batch=[6109], ptr=[257]), DataBatch(x=[6109, 2], edge_index=[2, 9992], edge_attr=[9992, 2], batch=[6109], ptr=[257])]
        #     #修改后：[DataBatch(x=[6407, 9], edge_index=[2, 10484], edge_attr=[10484, 2], batch=[6407], ptr=[257]), DataBatch(x=[6407, 9], edge_index=[2, 10484], edge_attr=[10484, 2], batch=[6407], ptr=[257])]
        #     #pt文件读取：[DataBatch(x=[3586, 9], edge_index=[2, 5706], edge_attr=[5706, 3], y=[256], edge_distance=[5706], batch=[3586], ptr=[257]), DataBatch(x=[3586, 9], edge_index=[2, 5706], edge_attr=[5706, 3], y=[256], edge_distance=[5706], batch=[3586], ptr=[257])]
        #     # 如果批次中包含多个图数据
        #     for i, data in enumerate(batch):
        #         print(f"Graph {i} information:")
        #         print(f"  x (node features):{data.x.shape}, {data.x}")
        #         print(f"  y :{data.y.shape}, {data.y}")
        #         print(f"  edge_distance:{data.edge_distance.shape}, {data.edge_distance}")
        #         print(f"  edge_index:{data.edge_index.shape}, {data.edge_index}")
        #         print(f"  edge_attr:{data.edge_attr.shape}, {data.edge_attr}")
        #         print(f"  batch (batch assignments):{data.batch.shape}, {data.batch}")
        #         print(f"  ptr (graph pointers):{data.ptr.shape}, {data.ptr}")
        #     break  # 打印一个批次的数据
        print("self.config[model]:",self.config["model"])
        if self.config['model_type'] == 'dignn':
            # from models.dignn import DIGNN, MERIT
            # model = DIGNN(**self.config["model"]).to(self.device)
            # model = self._load_pre_trained_weights(model)
            # merit = MERIT(self.config["model"], 0.8).to(self.device)
            # point_model = merit().to(self.device)
            pass

        elif self.config['model_type'] == 'gcn':
            # from models.gcn import GCN, MERIT
            # model = GCN(**self.config["model"]).to(self.device)
            # model = self._load_pre_trained_weights(model)
            # merit = MERIT(self.config["model"], 0.8).to(self.device)
            # point_model = merit().to(self.device)
            pass

        elif self.config['model_type'] == 'gin':
            # from models.ginet_molclr import GINet, MERIT
            # model = GINet(**self.config["model"]).to(self.device)
            # model = self._load_pre_trained_weights(model)
            # merit = MERIT(self.config["model"], 0.8).to(self.device)
            # point_model = merit().to(self.device)
            pass

        elif self.config['model_type'] == 'GS':
            from models.GS_ginet_molclr import GINet, MERIT
            model = GINet(**self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
            merit = MERIT(self.config["model"], 0.8).to(self.device)
            point_model = merit().to(self.device)

        else:
            raise ValueError('Undefined GNN model.')
        print(model)

        optimizer = torch.optim.Adam(
            model.parameters(), self.config['init_lr'],
            weight_decay=eval(self.config['weight_decay'])
        )


        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.config['epochs'] - self.config['warm_up'],
            eta_min=0, last_epoch=-1
        )


        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
            )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):

            train_losses = AverageMeter()
            train_1_losses = AverageMeter()
            train_2_losses = AverageMeter()
            train_3_losses = AverageMeter()
            train_4_losses = AverageMeter()
            train_5_losses = AverageMeter()
            train_6_losses = AverageMeter()
            for bn, (xis, xjs) in enumerate(tqdm(train_loader, desc="Training", ncols=80)):

                optimizer.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss_1 = self._step(model, xis, xjs, n_iter)
                loss_2 = self._step_n2(point_model, xis, xjs, n_iter)
                loss_3 = self._step_n1n2(model, point_model, xis, xis, n_iter)
                loss_4 = self._step_n1n2(model, point_model, xjs, xjs, n_iter)
                loss_5 = self._step_n1n2(model, point_model, xjs, xis, n_iter)
                loss_6 = self._step_n1n2(model, point_model, xis, xjs, n_iter)

                total_loss = 0.1 * (loss_1 + loss_5) + 0.4 * loss_2 + 0.2 * (loss_3 + loss_4)

                train_losses.update(total_loss.item(), self.config['batch_size'])
                train_1_losses.update(loss_1.item(), self.config['batch_size'])
                train_2_losses.update(loss_2.item(), self.config['batch_size'])
                train_3_losses.update(loss_3.item(), self.config['batch_size'])
                train_4_losses.update(loss_4.item(), self.config['batch_size'])
                train_5_losses.update(loss_5.item(), self.config['batch_size'])
                train_6_losses.update(loss_6.item(), self.config['batch_size'])

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', total_loss, global_step=n_iter)
                    self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                    print(f'\nepoch counter:{epoch_counter} batch num:{bn}')
                    print(f'loss1:{loss_1.item()}')
                    print(f'loss2:{loss_2.item()}')
                    print(f'loss3:{loss_3.item()}')
                    print(f'loss4:{loss_4.item()}')
                    print(f'loss5:{loss_5.item()}')
                    print(f'loss6:{loss_6.item()}')
                    print(f'total loss:{train_losses.avg}\n')

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    total_loss.backward()
                    optimizer.step()
                    merit.update_ma()

                # optimizer.step()

                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, point_model, valid_loader)

                print(f'epoch counter:{epoch_counter} batch num:{bn}')
                print(f'validate loss:{valid_loss}')

                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            if (epoch_counter + 1) % self.config['save_every_n_epochs'] == 0:
                torch.save(model.state_dict(),
                           os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(str(epoch_counter))))

            # warmup for the first few epochs
            if epoch_counter >= self.config['warm_up']:
                scheduler.step()

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['load_model'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, point_model, valid_loader):
        # validation steps
        with torch.no_grad():
            model.eval()
            point_model.eval()

            valid_loss = 0.0
            counter = 0
            for (xis, xjs) in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss_1 = self._step(model, xis, xjs, counter)
                loss_2 = self._step_n2(point_model, xis, xjs, counter)
                loss_3 = self._step_n1n2(model, point_model, xis, xis, counter)
                loss_4 = self._step_n1n2(model, point_model, xjs, xjs, counter)
                loss_5 = self._step_n1n2(model, point_model, xjs, xis, counter)
                loss_6 = self._step_n1n2(model, point_model, xis, xjs, counter)
                total_loss = 0.1 * (loss_1 + loss_5) + 0.7 * loss_2 + 0.2 * (loss_3 + loss_4)
                valid_loss += total_loss.item()
                counter += 1
            valid_loss /= counter

        model.train()
        point_model.train()

        return valid_loss


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    print(config)
    print(config['aug'])
    if config['aug'] == 'node':
        from dataset.dataset import MoleculeDatasetWrapper
    elif config['aug'] == 'subgraph':
        from dataset.dataset_subgraph import MoleculeDatasetWrapper
    elif config['aug'] == 'mix':
        from dataset.dataset_mix import MoleculeDatasetWrapper
    else:
        raise ValueError('Not defined molecule augmentation!')

    dataset = MoleculeDatasetWrapper(config['batch_size'], **config['dataset'])
    dig = DIG_Mol(dataset, config)
    dig.train()


if __name__ == "__main__":
    main()