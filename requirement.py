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


class MemN2N(nn.Module):
    """End-To-End Memory Network."""
    def __init__(self, embedding_size, key_size, mem_size, 
                 meta_size, hops=3, nonlin=None, name='MemN2N'):
        """
        initialize and declare variables
        """
        super().__init__()
        self._embedding_size = embedding_size
        self._embedding_size_x2 = embedding_size * 2
        self._mem_size = mem_size
        self._meta_size = meta_size
        self._key_size = key_size
        self._hops = hops
        self._nonlin = nonlin
        self._name = name

        self._queries = nn.Parameter(torch.normal(mean=0.0, std=0.01, 
                size=(1, self._embedding_size)), 
                requires_grad=True)
        self._A = nn.Parameter(torch.normal(mean=0.0, std=0.01, 
                size=(self._embedding_size, self._embedding_size_x2)), 
                requires_grad=True)
        self._B = nn.Parameter(torch.normal(mean=0.0, std=0.01, 
                size=(self._embedding_size, self._embedding_size_x2)), 
                requires_grad=True)
        self._C = nn.Parameter(torch.normal(mean=0.0, std=0.01, 
                size=(self._embedding_size, self._embedding_size_x2)), 
                requires_grad=True)
        self._H = nn.Parameter(torch.normal(mean=0.0, std=0.01, 
                size=(self._embedding_size_x2, self._embedding_size_x2)), 
                requires_grad=True)
        self._W = nn.Parameter(torch.normal(mean=0.0, std=0.01, 
                size=(self._embedding_size_x2, self._key_size)), 
                requires_grad=True)
            
    def forward(self, stories):
        """
        build graph for end-to-end memory network
        """
        # query embedding
        u_0 = torch.matmul(self._queries, self._B)
        u = [u_0]
        for _ in range(self._hops):
            # key embedding
            m_temp = torch.matmul(torch.reshape(stories, 
                        (-1, self._embedding_size)), self._A)
            m = torch.reshape(m_temp, 
                        (-1, self._mem_size, self._embedding_size_x2))
            u_temp = torch.transpose(
                        torch.unsqueeze(u[-1], -1), 2, 1)
            # get attention
            dotted = torch.sum(m * u_temp, 2) 
            probs = F.softmax(dotted, 1)
            probs_temp = torch.transpose(
                            torch.unsqueeze(probs, -1), 2, 1)
            # value embedding
            c = torch.matmul(torch.reshape(stories, 
                            (-1, self._embedding_size)), self._C)
            c = torch.reshape(c, 
                    (-1, self._mem_size, self._embedding_size_x2))
            c_temp = torch.transpose(c, 2, 1)
            # get intermediate result 
            o_k = torch.sum(c_temp * probs_temp, 2)
            u_k = torch.matmul(u[-1], self._H) + o_k
            if self._nonlin:
                u_k = self._nonlin(u_k)
            u.append(u_k)
        # get final result    
        req = torch.matmul(u[-1], self._W)    
        return req
            

class RequirementNet(nn.Module):
    """Requirement Network"""
    def __init__(self, emb_size, key_size, mem_size, meta_size, 
                 hops, name='RequirementNet'):
        """
        initialize and declare variables
        """
        super().__init__()
        self._name = name
        self._memn2n = MemN2N(emb_size, key_size, mem_size, meta_size, hops)

    def forward(self, dlg):
        """
        build graph for requirement estimation
        """
        req = self._memn2n(dlg)
       
        return req