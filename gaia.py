'''
AI Fashion Coordinator
(Baseline For Fashion-How Challenge)

MIT License

Copyright (C) 2023, Integrated Intelligence Research Section, ETRI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Update: 2023.02.15.
'''


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import timeit
import re
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from scipy import stats
from file_io import *
from requirement import *
from policy import *
from si import *
import joblib
import shutil
import time
import copy

# of items in fashion coordination      
NUM_ITEM_IN_COORDI = 4
# of metadata features    
NUM_META_FEAT = 4
# of fashion coordination candidates        
NUM_RANKING = 3
# image feature size 
IMG_FEAT_SIZE = 4096
# SI parameter
# si_c = 2.0
epsilon = 0.001

class Model(nn.Module):
    """ Model for AI fashion coordinator """
    def __init__(self, emb_size, key_size, mem_size, 
                 meta_size, hops, item_size, 
                 coordi_size, eval_node, num_rnk, 
                 use_batch_norm, use_dropout, zero_prob,
                 use_multimodal, img_feat_size):
        """
        initialize and declare variables
        """
        super().__init__()
        # class instance for requirement estimation
        self._requirement = RequirementNet(emb_size, key_size, 
                                    mem_size, meta_size, hops)
        # class instance for ranking
        self._policy = PolicyNet(emb_size, key_size, item_size, 
                                 meta_size, coordi_size, eval_node,
                                 num_rnk, use_batch_norm,
                                 use_dropout, zero_prob,
                                 use_multimodal, img_feat_size)
        
    def forward(self, dlg, crd):
        """
        build graph
        """
        req = self._requirement(dlg)
        logits = self._policy(req, crd)
        
        if self.training:
            return logits
        else:
            preds = torch.argmax(logits, dim=-1)
            return preds


class gAIa(object):
    """ Class for AI fashion coordinator """
    def __init__(self, args, device, name='gAIa'):
        """
        initialize
        """
        self._device = device
        self._batch_size = args.batch_size
        self._model_path = args.model_path
        self._model_file = args.model_file
        self._epochs = args.epochs
        self._max_grad_norm = args.max_grad_norm
        self._save_freq = args.save_freq
        self._num_eval = args.evaluation_iteration
        self._in_file_trn_dialog = args.in_file_trn_dialog
        self._use_cl = args.use_cl
        use_dropout = args.use_dropout
        if args.mode == 'test':
            use_dropout = False
        self.ntask = args.ntask
        self.inference = args.inference
        
        # class instance for subword embedding
        self._swer = SubWordEmbReaderUtil(args.subWordEmb_path)
        self._emb_size = self._swer.get_emb_size()
        self._meta_size = NUM_META_FEAT
        self._coordi_size = NUM_ITEM_IN_COORDI
        self._num_rnk = NUM_RANKING
        feats_size = IMG_FEAT_SIZE
        
        if self.ntask == 1:
            self._epochs = 40
            self.si = 0.0
            self.ce = 1.0
        if self.ntask == 2:
            self.si = 0.5
            self.ce = 1.2
        if self.ntask == 3:
            self.si = 1.0
            self.ce = 1.0
            args.learning_rate = 1e-3
        if self.ntask == 4:
            self.si = 0.5
            self.ce = 1.0
            args.learning_rate = 1e-3
        if self.ntask == 5:
            self.si = 1.0
            self.ce = 0.9
            args.learning_rate = 1e-3
            
        if self.ntask == 6:
            self.si = 1.0
            self.ce = 0.2
            args.learning_rate = 1e-3
        
        # read metadata DB
        self._metadata, self._idx2item, self._item2idx, \
            self._item_size, self._meta_similarities, \
            self._feats = make_metadata(args.in_file_fashion, self._swer, 
                                self._coordi_size, self._meta_size,
                                args.use_multimodal, args.in_file_img_feats,
                                feats_size)
        
        # prepare DB for training
        if args.mode == 'train':
            self._dlg, self._crd, self._rnk = make_io_data('prepare', 
                        args.in_file_trn_dialog, self._swer, args.mem_size,
                        self._coordi_size, self._item2idx, self._idx2item, 
                        self._metadata, self._meta_similarities, self._num_rnk,
                        args.permutation_iteration, args.num_augmentation, 
                        args.corr_thres, self._feats)
            self._num_examples = len(self._dlg)
            # dataloader
            dataset = TensorDataset(torch.tensor(self._dlg), 
                                    torch.tensor(self._crd), 
                                    torch.tensor(self._rnk, dtype=torch.long))
            self._dataloader = DataLoader(dataset, 
                                    batch_size=self._batch_size, shuffle=True)
        # prepare DB for evaluation
        elif args.mode in ['test', 'pred'] :
            self._tst_dlg, self._tst_crd, _ = make_io_data('eval', 
                    args.in_file_tst_dialog, self._swer, args.mem_size,
                    self._coordi_size, self._item2idx, self._idx2item, 
                    self._metadata, self._meta_similarities, self._num_rnk,
                    args.num_augmentation, args.num_augmentation, 
                    args.corr_thres, self._feats)
            self._num_examples = len(self._tst_dlg)
        
        print('\n<val data>')
        self._tst_dlgs = []
        self._tst_crds = []
        for i in range(1, self.ntask+1):
            dlg_path = f'./data/cl_eval_task{i}.wst.dev'
            _tst_dlg, _tst_crd, _ = make_io_data('eval',
                    dlg_path, self._swer, args.mem_size,
                    self._coordi_size, self._item2idx, self._idx2item,
                    self._metadata, self._meta_similarities, self._num_rnk,
                    args.num_augmentation, args.num_augmentation,
                    args.corr_thres, self._feats)
            self._tst_dlgs.append(_tst_dlg)
            self._tst_crds.append(_tst_crd)
        
        # model
        self._model = Model(self._emb_size, args.key_size, args.mem_size, 
                            self._meta_size, args.hops, self._item_size, 
                            self._coordi_size, args.eval_node, self._num_rnk, 
                            args.use_batch_norm, use_dropout, 
                            args.zero_prob, args.use_multimodal, 
                            feats_size)
        if not self.inference:
            print('\n<model parameters>')
            for name, param in self._model.named_parameters():
                if param.requires_grad:
                    print(name)
                    n = name.replace('.', '__')
                    self._model.register_buffer('{}_SI_prev_task'.format(n), 
                                                param.detach().clone())
                    self._model.register_buffer('{}_SI_omega'.format(n), 
                                                torch.zeros(param.shape))

        if args.mode == 'train':
            # optimizer
            self._optimizer = optim.SGD(self._model.parameters(), lr=args.learning_rate)
            # loss function
            self._criterion = nn.CrossEntropyLoss()

    def _get_loss(self, batch):
        """
        calculate loss
        """
        dlg, crd, rnk = batch
        logits= self._model(dlg, crd)
        ce_loss = self._criterion(logits, rnk)
        return ce_loss


    def train(self):
        """
        training
        """
        print('\n<Train>')
        print('total examples in dataset: {}'.format(self._num_examples))

        if not os.path.exists(self._model_path):
            os.makedirs(self._model_path)
        
        init_epoch = 1
        if self._model_file is not None:
            file_name = os.path.join(self._model_path, self._model_file)
            if os.path.exists(file_name):
                checkpoint = torch.load(file_name, 
                                        map_location=torch.device(self._device))
                self._model.load_state_dict(checkpoint['model'])
                print('[*] load success: {}\n'.format(file_name))
                # if the loaded model is the final model of the previous task, 
                # then backup the model
                # if self._model_file == 'gAIa-final.pt':
                #     print('time.strftime: ')
                #     print(time.strftime("%m%d-%H%M%S"))
                #     file_name_backup = os.path.join(self._model_path, 
                #         'gAIa-final-{}.pt'.format(time.strftime("%m%d-%H%M%S")))
                #     print('file_name_backup: ')
                #     print(file_name_backup)
                #     shutil.copy(file_name, file_name_backup)
                # else, start training from the loaded model
                # else:
                #     init_epoch += int(re.findall('\d+', file_name)[-1])
            else:
                print('[!] checkpoints path does not exist...\n')
                return False
        
        self._model.to(self._device)
        
        W = {}
        p_old = {}
        for n, p in self._model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                W[n] = p.data.clone().zero_()
                p_old[n] = p.data.clone()
        
        end_epoch = self._epochs + init_epoch
        val = 0
        f = open('log.txt', 'a')
        f.write(f"\nTask{self.ntask}*********************************************")
        for curr_epoch in range(init_epoch, end_epoch):
            # time_start = timeit.default_timer()
            losses = []
            si_losses = []
            iter_bar = tqdm(self._dataloader)
            
            for batch in iter_bar:
                self._optimizer.zero_grad()
                batch = [t.to(self._device) for t in batch]
                
                loss = self._get_loss(batch)
                loss_si = surrogate_loss(self._model) * self.si
                loss = loss*self.ce + loss_si
                loss.backward()
                # nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
                
                self._optimizer.step()
                losses.append(loss)
                si_losses.append(loss_si)
                for n, p in self._model.named_parameters():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        if p.grad is not None:
                            W[n].add_(-p.grad*(p.detach()-p_old[n]))
                        p_old[n] = p.detach().clone()
            # time_end = timeit.default_timer()
            task_list = [] # 각 테스크 성능.
            for TASK in range(self.ntask):
                max_per = self.test_during_train(TASK+1).item()
                task_list.append(max_per)
            task_list = torch.tensor(task_list)
            # print('-'*50)
            print('Epoch: {}/{}'.format(curr_epoch, end_epoch - 1))
            # print('Time: {:.2f}sec'.format(time_end - time_start))
            # print('Loss: {:.4f}'.format(torch.mean(torch.tensor(losses))))
            print('Total Loss: {:.4f}'.format(torch.mean(torch.tensor(losses))),
                    'SI loss: {:.4f}'.format(torch.mean(torch.tensor(si_losses))))
            current_lr = self._optimizer.param_groups[0]['lr']
            print(f'Learning Rate: {current_lr}')
            print('-'*50)
            print(torch.mean(task_list), val)
            f.write(f"\n\nEpoch {curr_epoch}/{end_epoch - 1}")
            f.write(f"\nTotal Loss {torch.mean(torch.tensor(losses)).cpu().numpy():.4f}")
            f.write(f"\nSI Loss {torch.mean(torch.tensor(si_losses)).cpu().numpy():.4f}")
            f.write(f"\nCurrent val score {torch.mean(task_list).cpu().numpy():.4f}")
            if val != 0:
                f.write(f"\nMax val score {val.cpu().numpy():.4f}")
            
            if (torch.mean(task_list) > val): # ntask 6
                f.write("save...")
                cur_model = copy.deepcopy(self._model)
                update_omega(cur_model, self._device, W, epsilon)
                val = torch.mean(task_list)
                file_name_final = os.path.join(self._model_path, 'gAIa-final.pt')
                torch.save({'model': cur_model.state_dict()}, file_name_final)
                
                file_name_final = os.path.join(self._model_path, 'gAIa-{}.pt'.format(self.ntask))
                torch.save({'model': cur_model.state_dict()}, file_name_final)
                
        print('Done training; epoch limit {} reached.\n'.format(self._epochs))
    
        f.close()
        
        # update_omega(self._model, self._device, W, epsilon)

        # file_name_final = os.path.join(self._model_path, 'gAIa-final.pt')
        # torch.save({'model': self._model.state_dict()}, file_name_final)
        
        # file_name_final = os.path.join(self._model_path, 'gAIa-{}.pt'.format(self.ntask))
        # torch.save({'model': self._model.state_dict()}, file_name_final)
        return True
        
    def _calculate_weighted_kendall_tau(self, pred, label, rnk_lst):
        """
        calcuate Weighted Kendall Tau Correlation
        """
        total_count = 0
        total_corr = 0
        for p, l in zip(pred, label):
            corr, _ = stats.weightedtau(self._num_rnk-1-rnk_lst[l], #
                                        self._num_rnk-1-rnk_lst[p]) #
            total_corr += corr
            total_count += 1
        return (total_corr / total_count)
    
    def _predict(self, eval_dlg, eval_crd):
        """
        predict
        """
        eval_num_examples = eval_dlg.shape[0]
        eval_dlg = torch.tensor(eval_dlg).to(self._device)
        eval_crd = torch.tensor(eval_crd).to(self._device)
        preds = []
        for start in range(0, eval_num_examples, self._batch_size):
            end = start + self._batch_size
            if end > eval_num_examples:
                end = eval_num_examples
            pred = self._model(eval_dlg[start:end],
                                  eval_crd[start:end])
            pred = pred.cpu().numpy()
            for j in range(end-start):
                preds.append(pred[j])
        preds = np.array(preds)
        return preds, eval_num_examples
    
    def _evaluate(self, eval_dlg, eval_crd):
        """
        evaluate
        """
        self._model.eval()
        eval_num_examples = eval_dlg.shape[0]
        eval_corr = []
        rank_lst = np.array(list(permutations(np.arange(self._num_rnk), 
                                              self._num_rnk)))         
        eval_dlg = torch.tensor(eval_dlg).to(self._device)
        repeated_preds = []
        with torch.no_grad():
            for i in range(self._num_eval):
                preds = []
                # DB shuffling
                coordi, rnk = shuffle_coordi_and_ranking(eval_crd, self._num_rnk)
                coordi = torch.tensor(coordi).to(self._device)
                for start in range(0, eval_num_examples, self._batch_size):
                    end = start + self._batch_size
                    if end > eval_num_examples:
                        end = eval_num_examples
                    pred = self._model(eval_dlg[start:end], 
                                        coordi[start:end])
                    pred = pred.cpu().numpy()
                    for j in range(end-start):
                        preds.append(pred[j])
                preds = np.array(preds)
                # compute Weighted Kendall Tau Correlation
                repeated_preds.append(preds)
                corr = self._calculate_weighted_kendall_tau(preds, rnk, rank_lst)
                eval_corr.append(corr)
        return repeated_preds, np.array(eval_corr), eval_num_examples
    
    def pred(self):
        """
        create prediction.csv
        """
        print('\n<Predict>')
        self._model.eval()
        if self._model_file is not None:
            file_name = os.path.join(self._model_path, self._model_file)
            if os.path.exists(file_name):
                from torch.quantization import quantize_dynamic
                self._model.to('cpu')
                checkpoint = torch.load(file_name)
                _model_quant = quantize_dynamic(self._model, {nn.Linear}, dtype=torch.qint8)
                _model_quant.load_state_dict(checkpoint['model'])
                self._model = _model_quant
                # self._model.to(self._device)
                print('[*] load success: {}\n'.format(file_name))
            else:
                print('[!] checkpoints path does not exist...\n')
                return False
        else:
            return False
        time_start = timeit.default_timer()
        # predict
        preds, num_examples = self._predict(self._tst_dlg, self._tst_crd)
        time_end = timeit.default_timer()
        print('-'*50)
        print('Prediction Time: {:.2f}sec'.format(time_end-time_start))
        print('# of Test Examples: {}'.format(num_examples))
        print('-'*50)
        return preds.astype(int)

    def test(self):
        """
        get scores using sample test set
        """
        print('\n<Evaluate>')
        self._model.eval()
        if self._model_file is not None:
            file_name = os.path.join(self._model_path, self._model_file)
            if os.path.exists(file_name):
                # from torch.quantization import quantize_dynamic
                checkpoint = torch.load(file_name, 
                                        map_location=torch.device(self._device))
                self._model.load_state_dict(checkpoint['model'],strict=False)
                self._model.to(self._device)
                
                if (self.ntask == 6) & (self.inference==True):                    
                    from torch.quantization import quantize_dynamic
                    self._model.to('cpu')
                    _model_quant = quantize_dynamic(self._model, {nn.Linear}, dtype=torch.qint8)
                    file_name_final = os.path.join(self._model_path, 'gAIa-final.pt')
                    torch.save({'model': _model_quant.state_dict()}, file_name_final)
                    self._model.to(self._device)
                    
                    
                
                print('[*] load success: {}\n'.format(file_name))
            else:
                print('[!] checkpoints path does not exist...\n')
                return False
        else:
            return False
        time_start = timeit.default_timer()
        # evaluation
        repeated_preds, test_corr, num_examples = self._evaluate(self._tst_dlg, 
                                                                 self._tst_crd)
        time_end = timeit.default_timer()
        print('-'*50)
        print('Prediction Time: {:.2f}sec'.format(time_end-time_start))
        print('# of Test Examples: {}'.format(num_examples))
        print('Average WKTC over iterations: {:.4f}'.format(np.mean(test_corr)))
        print('Best WKTC: {:.4f}'.format(np.max(test_corr)))
        print('-'*50)
        
        f = open('log.txt','a')
        f.write(f"\nTask{self.ntask} validation***************************************")
        f.write(f"\n Average WKTC over iterations {np.mean(test_corr):.4f}")
        f.close()
        return np.mean(test_corr)
    
    def test_during_train(self, task):
        """
        get scores using sample test set
        """
        print(f'\n<Evaluate task: {task}>')
        self._model.eval()
        time_start = timeit.default_timer()
        # evaluation
        repeated_preds, test_corr, num_examples = self._evaluate(self._tst_dlgs[task-1], self._tst_crds[task-1])
        time_end = timeit.default_timer()
        print('-'*50)
        # print('Prediction Time: {:.2f}sec'.format(time_end-time_start))
        # print('# of Test Examples: {}'.format(num_examples))
        print('Average WKTC over iterations: {:.4f}'.format(np.mean(test_corr)))
        print('Best WKTC: {:.4f}'.format(np.max(test_corr)))
        print('-'*50)
        self._model.train()
        return np.mean(test_corr)
