import os
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, roc_auc_score

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CyclicLR
from torch.nn.utils.rnn import pad_sequence

from utils.utils import (Lamb, Lookahead, RobustL1, BCEWithLogitsLoss,
                         EDM_CsvLoader, Scaler, DummyScaler, count_parameters)
from utils.get_compute_device import get_compute_device
from utils.optim import SWA

import sys
sys.path.append("/home/zd/zd/teaching_net/alignn/alignn/")
from pretrained import get_multiple_predictions, get_figshare_model
from data import get_torch_dataset
from torch.utils.data import DataLoader


# %%
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32


# %%
class Model():
    def __init__(self,
                 model,
                 model_name='UnnamedModel',
                 n_elements='infer',
                 capture_every=None,
                 verbose=True,
                 drop_unary=True,
                 scale=True,
                 embedding_loss=False,
                 save_path=None,
                 pretrained=False,
                 finetune = False,
                 embedding_path=None,
                 transformer_model=False,
                 crab_ori_embedding=False,):
        self.model = model
        self.model_name = model_name
        self.save_path = save_path
        self.data_loader = None
        self.train_loader = None
        self.classification = False
        self.n_elements = n_elements
        self.compute_device = model.compute_device
        self.fudge = 0.02  #  expected fractional tolerance (std. dev) ~= 2%
        self.capture_every = capture_every
        self.verbose = verbose
        self.drop_unary = drop_unary
        self.scale = scale
        if self.compute_device is None:
            self.compute_device = get_compute_device()
        self.capture_flag = False
        self.formula_current = None
        self.act_v = None
        self.pred_v = None
        if self.verbose:
            print('\nModel architecture: out_dims, d_model, N, heads')
            print(f'{self.model.out_dims}, {self.model.d_model}, '
                  f'{self.model.N}, {self.model.heads}')
            print(f'Running on compute device: {self.compute_device}')
            print(f'Model size: {count_parameters(self.model)} parameters\n')
        if self.capture_every is not None:
            print(f'capturing attention tensors every {self.capture_every}')
        
        self.embedding_loss = embedding_loss
        self.pretrain = pretrained
        self.finetune = finetune
        self.transformer_model = transformer_model
        self.crab_ori_embedding = crab_ori_embedding
        self.criterion_transformer = nn.MSELoss(reduction='none')
        if embedding_loss or transformer_model and embedding_path is not None:
            import pickle
            # with open('/home/zd/zd/teaching_net/alignn/alignn/data_formula_atom/mp_e_form/real_ma_e_form/mp_e_form/mp_eval_formular_embedding.pkl', 'rb') as f:
            with open(embedding_path, 'rb') as f:
                csv_atoms = pickle.load(f)
            self.csv_atoms = csv_atoms
        if self.crab_ori_embedding:
            import pickle
            # with open('/home/zd/zd/teaching_net/alignn/alignn/data_formula_atom/mp_e_form/real_ma_e_form/mp_e_form/mp_eval_formular_embedding.pkl', 'rb') as f:
            with open('/home/zd/zd/teaching_net/CrabNet/predict_structurEmbedding/jarvis_e_form/embeddings/jarvis_crabnet_oriEmbedding.pkl', 'rb') as f:
                ori_embedding = pickle.load(f)
            self.ori_embedding = ori_embedding

    def load_data(self, file_name, batch_size=2**9, train=False):
        self.batch_size = batch_size
        inference = not train
        data_loaders = EDM_CsvLoader(csv_data=file_name,
                                     batch_size=batch_size,
                                     n_elements=self.n_elements,
                                     inference=inference,
                                     verbose=self.verbose,
                                     drop_unary=self.drop_unary,
                                     scale=self.scale,
                                     ratio=1)   # ratio is the sample ratio of the used dataset
        print(f'loading data with up to {data_loaders.n_elements:0.0f} '
              f'elements in the formula')

        # update n_elements after loading dataset
        self.n_elements = data_loaders.n_elements

        data_loader = data_loaders.get_data_loaders(inference=inference)
        y = data_loader.dataset.data[1]
        if train:
            self.train_len = len(y)
            if self.classification:
                self.scaler = DummyScaler(y)
            else:
                self.scaler = Scaler(y)
            self.train_loader = data_loader
        self.data_loader = data_loader


    def train(self):
        self.model.train()
        ti = time()
        minima = []
        for i, data in enumerate(self.train_loader):
            X, y, formula = data
            y = self.scaler.scale(y)
            # print('x shape: {0}'.format(X.shape))
            src, frac = X.squeeze(-1).chunk(2, dim=1)
            # add a small jitter to the input fractions to improve model
            # robustness and to increase stability
            # frac = frac * (1 + (torch.rand_like(frac)-0.5)*self.fudge)  # uniform
            frac = frac * (1 + (torch.randn_like(frac))*self.fudge)  # normal
            frac = torch.clamp(frac, 0, 1)
            frac[src == 0] = 0
            frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])

            src = src.to(self.compute_device,
                         dtype=torch.long,
                         non_blocking=True)
            frac = frac.to(self.compute_device,
                           dtype=data_type_torch,
                           non_blocking=True)
            y = y.to(self.compute_device,
                     dtype=data_type_torch,
                     non_blocking=True)

            ##################################
            # Force evaluate dataset so that we can capture it in the hook
            # here we are using the train_loader, but we can also use
            # general data_loader
            if self.capture_every == 'step':
                # print('capturing every step!')
                # print(f'data_loader size: {len(self.data_loader.dataset)}')
                self.capture_flag = True
                # (act, pred, formulae, uncert)
                self.act_v, self.pred_v, _, _, _ = self.predict(self.data_loader)
                self.capture_flag = False
            ##################################

            if self.transformer_model:
                # trg = [self.csv_atoms[formu] for formu in formula]
                # # trg = [torch.from_numpy(self.csv_atoms[formu]) for formu in formula]
                if torch.is_tensor(self.csv_atoms[formula[0]]):
                    trg = [self.csv_atoms[formu] for formu in formula]
                else:
                    trg = [torch.from_numpy(self.csv_atoms[formu]).unsqueeze(0) for formu in formula]
                trg = pad_sequence(trg, batch_first=True).to(self.compute_device)

                #use class token
            

                # print(trg.shape)
                # do not use this method because I only use the encoder for the property prediction
                # trg = torch.cat([torch.ones(trg.shape[0],1,trg.shape[2]).to(trg.device),\
                # trg, torch.ones(trg.shape[0],1,trg.shape[2]).to(trg.device)], dim=1)
                # trg = torch.cat([torch.zeros(trg.shape[0],trg.shape[1],1).to(trg.device), trg], dim=2)
                # trg[:,0,0] = 1; trg[:,-1,0] = -1

                trg_pad_mask = trg.eq(0).all(dim=-1)
                # print(trg_pad_mask.shape, trg.shape)
                output, embs_carbnet_transformer = self.model.forward(src, frac, trg,\
                                                                       trg_pad_mask=trg_pad_mask)
                if not self.finetune:
                    trg_pad_mask = trg_pad_mask.unsqueeze(-1).expand_as(trg)
                    predictions = embs_carbnet_transformer * ~trg_pad_mask
                    trg_masked = trg * ~trg_pad_mask
                    loss = self.criterion_transformer(predictions, trg_masked)
                    loss = loss * ~trg_pad_mask
                    loss = loss.sum() / (~trg_pad_mask).sum()
                    print(loss)
                else:
                    prediction = output
                    loss = self.criterion(prediction.view(-1),
                                        None,
                                        y.view(-1))

            elif self.crab_ori_embedding:
                ori_emb = [torch.from_numpy(self.ori_embedding[formu]).unsqueeze(0) for formu in formula]
                ori_emb = torch.cat(ori_emb, dim=0).to(self.compute_device)
                output, embs_carbnet = self.model.forward(src, frac, crab_ori=ori_emb)
                # print(output.shape)
                # prediction, uncertainty = output.chunk(2, dim=-1)
                # loss = self.criterion(prediction.view(-1),
                #                       uncertainty.view(-1),
                #                       y.view(-1))
                prediction = output
                loss = self.criterion(prediction.view(-1),
                                    None,
                                    y.view(-1))
            else:
                output, embs_carbnet = self.model.forward(src, frac)
                # print(output.shape)
                # prediction, uncertainty = output.chunk(2, dim=-1)
                # loss = self.criterion(prediction.view(-1),
                #                       uncertainty.view(-1),
                #                       y.view(-1))
                prediction = output
                loss = self.criterion(prediction.view(-1),
                                    None,
                                    y.view(-1))
            
            if self.embedding_loss:
                if torch.is_tensor(self.csv_atoms[formula[0]]):
                    # batch_embedding = [self.csv_atoms[formu] for formu in formula]
                    batch_embedding = [self.csv_atoms[formu].unsqueeze(0) for formu in formula]
                else:
                    batch_embedding = [torch.from_numpy(self.csv_atoms[formu]).unsqueeze(0) for formu in formula]
                # print(batch_embedding[0].shape, embs_carbnet.shape)
                embs_alignn = torch.cat(batch_embedding, dim=0).to(self.compute_device)
                # print(embs_alignn.shape)
                # embs_alignn = torch.cat(self.calculate_embedding(formula, device=output.device), dim=0).to(self.compute_device)
                loss_emb = self.criterion_emb(embs_carbnet.squeeze(), embs_alignn.squeeze())
                alpha = 20      # default 6
                beta = 1
                if self.pretrain and not self.finetune:
                    beta = 0
               
                print(loss, loss_emb)
                loss = (loss*beta + loss_emb*alpha)/(beta+alpha)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.stepping:
                self.lr_scheduler.step()

            swa_check = (self.epochs_step * self.swa_start - 1)
            epoch_check = (self.epoch + 1) % (2 * self.epochs_step) == 0
            learning_time = epoch_check and self.epoch >= swa_check
            if learning_time:
                with torch.no_grad():
                    act_v, pred_v, _, _, _ = self.predict(self.data_loader)
                mae_v = mean_absolute_error(act_v, pred_v)
                self.optimizer.update_swa(mae_v)
                minima.append(self.optimizer.minimum_found)

        if learning_time and not any(minima):
            self.optimizer.discard_count += 1
            print(f'Epoch {self.epoch} failed to improve.')
            print(f'Discarded: {self.optimizer.discard_count}/'
                  f'{self.discard_n} weight updates ♻🗑️')

        dt = time() - ti
        datalen = len(self.train_loader.dataset)
        # print(f'training speed: {datalen/dt:0.3f}')


    def fit(self, epochs=None, checkin=None, losscurve=False):
        assert_train_str = 'Please Load Training Data (self.train_loader)'
        assert_val_str = 'Please Load Validation Data (self.data_loader)'
        assert self.train_loader is not None, assert_train_str
        assert self.data_loader is not None, assert_val_str
        self.loss_curve = {}
        self.loss_curve['train'] = []
        self.loss_curve['val'] = []

        # change epochs_step
        # self.epochs_step = 10
        self.epochs_step = 1
        self.step_size = self.epochs_step * len(self.train_loader)
        print(f'stepping every {self.step_size} training passes,',
              f'cycling lr every {self.epochs_step} epochs')
        if epochs is None:
            n_iterations = 1e4
            epochs = int(n_iterations / len(self.data_loader))
            print(f'running for {epochs} epochs')
        if checkin is None:
            checkin = self.epochs_step * 2
            print(f'checkin at {self.epochs_step*2} '
                  f'epochs to match lr scheduler')
        if epochs % (self.epochs_step * 2) != 0:
            # updated_epochs = epochs - epochs % (self.epochs_step * 2)
            # print(f'epochs not divisible by {self.epochs_step * 2}, '
            #       f'updating epochs to {updated_epochs} for learning')
            updated_epochs = epochs
            epochs = updated_epochs

        self.step_count = 0
        self.criterion = RobustL1
        self.criterion_emb = nn.MSELoss()
        if self.classification:
            print("Using BCE loss for classification task")
            self.criterion = BCEWithLogitsLoss
        base_optim = Lamb(params=self.model.parameters())
        optimizer = Lookahead(base_optimizer=base_optim)
        self.optimizer = SWA(optimizer)

        # original learning rate
        # base_lr=1e-4,
        # max_lr=6e-3,        
                                #         base_lr=5e-5,
                                # max_lr=1e-3,

        lr_scheduler = CyclicLR(self.optimizer,
                                # base_lr=1e-4,
                                # max_lr=6e-3,
                                base_lr=1e-5,
                                max_lr=1e-3,                                
                                cycle_momentum=False,
                                step_size_up=self.step_size)

        self.swa_start = 2  # start at (n/2) cycle (lr minimum)
        self.lr_scheduler = lr_scheduler
        self.stepping = True
        self.lr_list = []
        self.xswa = []
        self.yswa = []
        # self.discard_n = 3
        # self.discard_n = 100
        self.discard_n = 150

        for epoch in range(epochs):
            self.epoch = epoch
            self.epochs = epochs
            ti = time()
            self.train()
            # print(f'epoch time: {(time() - ti):0.3f}')
            self.lr_list.append(self.optimizer.param_groups[0]['lr'])

            ##################################
            # Force evaluate dataset so that we can capture it in the hook
            # here we are using the train_loader, but we can also use
            # general data_loader
            if self.capture_every == 'epoch':
                # print('capturing every epoch!')
                # print(f'data_loader size: {len(self.data_loader.dataset)}')
                self.capture_flag = True
                # (act, pred, formulae, uncert)
                self.act_v, self.pred_v, _, _, _ = self.predict(self.data_loader)
                self.capture_flag = False
            ##################################

            if (epoch+1) % checkin == 0 or epoch == epochs - 1 or epoch == 0:
                ti = time()
                with torch.no_grad():
                    act_t, pred_t, _, _, _ = self.predict(self.train_loader)
                dt = time() - ti
                datasize = len(act_t)
                # print(f'inference speed: {datasize/dt:0.3f}')
                mae_t = mean_absolute_error(act_t, pred_t)
                self.loss_curve['train'].append(mae_t)
                with torch.no_grad():
                    act_v, pred_v, _, _, _ = self.predict(self.data_loader)
                mae_v = mean_absolute_error(act_v, pred_v)
                self.loss_curve['val'].append(mae_v)
                epoch_str = f'Epoch: {epoch}/{epochs} ---'
                train_str = f'train mae: {self.loss_curve["train"][-1]:0.3g}'
                val_str = f'val mae: {self.loss_curve["val"][-1]:0.3g}'
                if self.classification:
                    train_auc = roc_auc_score(act_t, pred_t)
                    val_auc = roc_auc_score(act_v, pred_v)
                    train_str = f'train auc: {train_auc:0.3f}'
                    val_str = f'val auc: {val_auc:0.3f}'
                print(epoch_str, train_str, val_str)

                if self.epoch >= (self.epochs_step * self.swa_start - 1):
                    if (self.epoch+1) % (self.epochs_step * 2) == 0:
                        self.xswa.append(self.epoch)
                        self.yswa.append(mae_v)

                if losscurve:
                    plt.figure(figsize=(8, 5))
                    xval = np.arange(len(self.loss_curve['val'])) * checkin - 1
                    xval[0] = 0
                    plt.plot(xval, self.loss_curve['train'],
                             'o-', label='train_mae')
                    plt.plot(xval, self.loss_curve['val'],
                             's--', label='val_mae')
                    plt.plot(self.xswa, self.yswa,
                             'o', ms=12, mfc='none', label='SWA point')
                    plt.ylim(0, 2 * np.mean(self.loss_curve['val']))
                    plt.title(f'{self.model_name}')
                    plt.xlabel('epochs')
                    plt.ylabel('MAE')
                    plt.legend()
                    plt.show()
            if epoch == epochs-1:
                self.save_network(f'{self.model_name}_epoch{epoch}')
                # torch.save(self.model.state_dict(),\
                #             "/home/zd/zd/teaching_net/CrabNet/predict_structurEmbedding/jarvis_bandgap/models/jarvis_bandgap_struPred_epoch299.pth")

            if (epoch == epochs-1 or
                self.optimizer.discard_count >= self.discard_n):
                # save output df for stats tracking
                xval = np.arange(len(self.loss_curve['val'])) * checkin - 1
                xval[0] = 0
                tval = self.loss_curve['train']
                vval = self.loss_curve['val']
                os.makedirs('figures/lc_data', exist_ok=True)
                df_loss = pd.DataFrame([xval, tval, vval]).T
                df_loss.columns = ['epoch', 'train loss', 'val loss']
                df_loss['swa'] = ['n'] * len(xval)
                df_loss.loc[df_loss['epoch'].isin(self.xswa), 'swa'] = 'y'
                df_loss.to_csv(f'figures/lc_data/{self.model_name}_lc.csv',
                               index=False)

                # save output learning curve plot
                plt.figure(figsize=(8, 5))
                xval = np.arange(len(self.loss_curve['val'])) * checkin - 1
                xval[0] = 0
                plt.plot(xval, self.loss_curve['train'],
                         'o-', label='train_mae')
                plt.plot(xval, self.loss_curve['val'], 's--', label='val_mae')
                if self.epoch >= (self.epochs_step * self.swa_start - 1):
                    plt.plot(self.xswa, self.yswa,
                             'o', ms=12, mfc='none', label='SWA point')
                plt.ylim(0, 2 * np.mean(self.loss_curve['val']))
                plt.title(f'{self.model_name}')
                plt.xlabel('epochs')
                plt.ylabel('MAE')
                plt.legend()
                plt.savefig(f'figures/lc_data/{self.model_name}_lc.png')

            if self.optimizer.discard_count >= self.discard_n:
                print(f'Discarded: {self.optimizer.discard_count}/'
                      f'{self.discard_n} weight updates, '
                      f'early-stopping now 🙅🛑')
                self.optimizer.swap_swa_sgd()
                break

        if not (self.optimizer.discard_count >= self.discard_n):
            self.optimizer.swap_swa_sgd()


    def predict(self, loader):
        len_dataset = len(loader.dataset)
        n_atoms = int(len(loader.dataset[0][0])/2)
        act = np.zeros(len_dataset)
        pred = np.zeros(len_dataset)
        uncert = np.zeros(len_dataset)
        formulae = np.empty(len_dataset, dtype=list)
        atoms = np.empty((len_dataset, n_atoms))
        fractions = np.empty((len_dataset, n_atoms))

        embeddings = {}
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                X, y, formula = data
                if self.capture_flag:
                    self.formula_current = None
                    # HACK for PyTorch v1.8.0
                    # this output used to be a list, but is now a tuple
                    if isinstance(formula, tuple):
                        self.formula_current = list(formula)
                    elif isinstance(formula, list):
                        self.formula_current = formula.copy()
                src, frac = X.squeeze(-1).chunk(2, dim=1)
                src = src.to(self.compute_device,
                             dtype=torch.long,
                             non_blocking=True)
                frac = frac.to(self.compute_device,
                               dtype=data_type_torch,
                               non_blocking=True)
                y = y.to(self.compute_device,
                         dtype=data_type_torch,
                         non_blocking=True)
                if self.transformer_model:
                    trg = [self.csv_atoms[formu] for formu in formula]
                    trg = pad_sequence(trg, batch_first=True).to(self.compute_device)
                    # trg = torch.cat([torch.ones(trg.shape[0],1,trg.shape[2]).to(trg.device),\
                    # trg, torch.ones(trg.shape[0],1,trg.shape[2]).to(trg.device)], dim=1)
                    # trg = torch.cat([torch.zeros(trg.shape[0],trg.shape[1],1).to(trg.device), trg], dim=2)
                    # trg[:,0,0] = 1; trg[:,-1,0] = -1

                    trg_pad_mask = trg.eq(0).all(dim=-1)
                    # print(trg_pad_mask.shape, trg.shape)
                    output, embedding = self.model.forward(src, frac, trg,\
                                                    trg_pad_mask=trg_pad_mask)
                elif self.crab_ori_embedding:
                    ori_emb = [torch.from_numpy(self.ori_embedding[formu]).unsqueeze(0) for formu in formula]
                    ori_emb = torch.cat(ori_emb, dim=0).to(self.compute_device)
                    output, embedding = self.model.forward(src, frac, crab_ori=ori_emb)
                    # print(output.shape)
                    # prediction, uncertainty = output.chunk(2, dim=-1)
                    # loss = self.criterion(prediction.view(-1),
                    #                       uncertainty.view(-1),
                    #                       y.view(-1))
                    # prediction = output
                    # loss = self.criterion(prediction.view(-1),
                    #                     None,
                    #                     y.view(-1))    
                else:
                    output, embedding = self.model.forward(src, frac)
                embeddings.update(dict(zip(formula, embedding.cpu().numpy())))
                # embeddings.update(dict(zip(formula, embedding.contiguous())))
                # prediction, uncertainty = output.chunk(2, dim=-1)
                # uncertainty = torch.exp(uncertainty) * self.scaler.std
                prediction = output
                uncertainty = torch.zeros_like(prediction)
                prediction = self.scaler.unscale(prediction)
                if self.classification:
                    prediction = torch.sigmoid(prediction)

                data_loc = slice(i*self.batch_size,
                                 i*self.batch_size+len(y),
                                 1)

                atoms[data_loc, :] = src.cpu().numpy().astype('int32')
                fractions[data_loc, :] = frac.cpu().numpy().astype('float32')
                act[data_loc] = y.view(-1).cpu().numpy().astype('float32')
                pred[data_loc] = prediction.view(-1).cpu().detach().numpy().astype('float32')
                uncert[data_loc] = uncertainty.view(-1).cpu().detach().numpy().astype('float32')
                formulae[data_loc] = formula
        self.model.train()
        # print(len(embeddings), embeddings[formula[0]].shape, embeddings[formula[0]].dtype)

        return (act, pred, formulae, uncert, embeddings)


    def save_network(self, model_name=None):
        if model_name is None:
            model_name = self.model_name
            os.makedirs(self.save_path, exist_ok=True)
            path = os.path.join(self.save_path, f'{model_name}.pth')
            print(f'Saving network ({model_name}) to {path}')
        else:
            path = os.path.join(self.save_path, f'{model_name}.pth')
            print(f'Saving checkpoint ({model_name}) to {path}')

        save_dict = {'weights': self.model.state_dict(),
                     'scaler_state': self.scaler.state_dict(),
                     'model_name': model_name}
        torch.save(save_dict, path)


    def load_network(self, path=None, finetune=False):
        if path is None:
            print(self.model_name, self.save_path)
            path = os.path.join(self.save_path, f'{self.model_name}.pth')
        else:
            path = path
        network = torch.load(path, map_location=self.compute_device)
        base_optim = Lamb(params=self.model.parameters())
        optimizer = Lookahead(base_optimizer=base_optim)
        self.optimizer = SWA(optimizer)
        self.scaler = Scaler(torch.zeros(3))
        if finetune:
            print(type(network['weights']))
            pretrained_state_dict = {k: v for k, v in network['weights'].items() if not k.startswith('decoder') and not k.startswith('transformer_decoder')}
            self.model.load_state_dict(pretrained_state_dict, strict=False)
            print('loaded modules: {}'.format(pretrained_state_dict.keys()))
        else:
            self.model.load_state_dict(network['weights'])
            # self.model.load_state_dict(network)
        self.scaler.load_state_dict(network['scaler_state'])
        # self.model_name = network['model_name']


    def calculate_embedding(self, formula, device):
        atoms_array = []
        # for formu in formula:
        #     atoms_array.append(self.csv_atoms[formu])
        atoms_array = [self.csv_atoms[formu] for formu in formula]
        mem = []
        for i, ii in enumerate(atoms_array):
            info = {}
            info["atoms"] = ii.to_dict()
            info["prop"] = -9999  # place-holder only
            info["jid"] = str(i)
            mem.append(info)
        test_data = get_torch_dataset(
            dataset=mem,
            target="prop",
            neighbor_strategy="k-nearest",
            atom_features="cgcnn",
            use_canonize=True,
            line_graph=True,
        )

        collate_fn = test_data.collate_line_graph
        test_loader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=4,
            pin_memory=False,
        )

        embs = []
        self.model_alignn.eval()
        with torch.no_grad():
            ids = test_loader.dataset.ids
            for dat, id in zip(test_loader, ids):
                g, lg, target = dat
                out_data = self.model_alignn([g.to(device), lg.to(device)])
                embs.append(out_data[1])
        return embs 

# %%
if __name__ == '__main__':
    pass
