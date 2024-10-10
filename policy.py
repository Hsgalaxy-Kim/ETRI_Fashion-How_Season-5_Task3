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

Update: 2022.06.16.
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict


class PolicyNet(nn.Module):
    """Class for policy network"""
    def __init__(self, emb_size, key_size, item_size, meta_size, 
                 coordi_size, eval_node, num_rnk, use_batch_norm, 
                 use_dropout, zero_prob, use_multimodal,
                 img_feat_size, name='PolicyNet'):
        """
        initialize and declare variables
        """
        super().__init__()
        self._item_size = item_size
        self._emb_size = emb_size
        self._key_size = key_size
        self._meta_size = meta_size
        self._coordi_size = coordi_size
        self._num_rnk = num_rnk
        self._use_dropout = use_dropout
        self._zero_prob = zero_prob
        self._name = name
        buf = eval_node[1:-1].split('][')
        self._num_hid_eval = list(map(int, buf[0].split(',')))
        self._num_hid_rnk = list(map(int, buf[1].split(',')))
        self._num_hid_layer_eval = len(self._num_hid_eval)
        self._num_hid_layer_rnk = len(self._num_hid_rnk)
        
        num_in = self._emb_size * self._meta_size * self._coordi_size + self._key_size

        self.layer1 = nn.Sequential(
            nn.Linear(num_in, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=self._zero_prob)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=self._zero_prob)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=self._zero_prob)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=self._zero_prob)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=self._zero_prob)
        )
        self.layer6 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=self._zero_prob)
        )
        self.layer7 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=self._zero_prob)
        )
        self.layer8 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=self._zero_prob)
        )
        self.layer9 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=self._zero_prob)
        )
        self.layer10 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=self._zero_prob)
        )
        self.layer11 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=self._zero_prob)
        )
        ######################################
        self.linear1 = nn.Sequential(
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=self._zero_prob)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=self._zero_prob)
        )
        self.linear3 = nn.Sequential(
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=self._zero_prob)
        )
        self.linear4 = nn.Sequential(
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=self._zero_prob)
        )
        self.linear5 = nn.Sequential(
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=self._zero_prob)
        )
        self.linear6 = nn.Linear(512,6)

    def _evaluate_coordi(self, crd, req):
        """
        evaluate candidates
        """
        crd_and_req = torch.cat((crd, req), 1)
        out1 = self.layer1(crd_and_req)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2 + out1)
        
        out4 = self.layer4(out3)
        out5 = self.layer5(out4+out3)
        
        out6 = self.layer6(out5)
        out7 = self.layer7(out6+out5)
        
        out8 = self.layer8(out7)
        out9 = self.layer9(out8 + out7)
        
        out10 = self.layer8(out9)
        out11 = self.layer9(out10 + out9)
        
        return out11
    
    def _ranking_coordi(self, in_rnk):
        """
        rank candidates         
        """
        lin1 = self.linear1(in_rnk)
        lin2 = self.linear2(lin1)
        lin3 = self.linear3(lin2 + lin1)
        lin4 = self.linear4(lin3)
        lin5 = self.linear5(lin4+lin3)
        lin6 = self.linear6(lin5)
        return lin6
        
    def forward(self, req, crd):
        """
        build graph for evaluation and ranking         
        crd [256,3,2048]
        """
        
        crd_tr = torch.transpose(crd, 1, 0)
        for i in range(self._num_rnk):
            crd_eval = self._evaluate_coordi(crd_tr[i], req)
            if i == 0:
                in_rnk = crd_eval
            else:
                in_rnk = torch.cat((in_rnk, crd_eval), 1)
        in_rnk = torch.cat((in_rnk, req), 1)        
        out_rnk = self._ranking_coordi(in_rnk)
        return out_rnk