"""
End-to-End neural ad-hoc ranking with kernel pooling
http://www.cs.cmu.edu/~zhuyund/papers/end-end-neural.pdf
5th HA of Ranking&Matching module of Hard ML specialization
"""


from typing import Dict, List

import numpy as np
import torch


class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(-0.5*torch.pow((x - self.mu), 2)/(self.sigma**2))

class KNRM(torch.nn.Module):
    """KNRM class
    """
    def __init__(self, embedding_matrix: np.ndarray, freeze_embeddings: bool, kernel_num: int = 21,
                 sigma: float = 0.1, exact_sigma: float = 0.001,
                 out_layers: List[int] = [10, 5]):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()
        self.mlp = self._get_mlp()
        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        """builds kernel layer
        """
        def get_mu():
            d = 2 / (self.kernel_num -1)
            D = (d*self.kernel_num)/2
            res = [D - d*k for k in range(self.kernel_num)]
            res = res[::-1]
            res[0] = max(res[0], -1)
            res[-1] = min(res[-1], 1)
            return res

        kernels = []
        k_mu = get_mu() 
        for i in range(self.kernel_num - 1):
            kernels.append(GaussianKernel(k_mu[i], self.sigma)) 
        kernels.append(GaussianKernel(k_mu[i+1], self.exact_sigma)) 

        return torch.nn.ModuleList(kernels)

    def _get_mlp(self) -> torch.nn.Sequential:
        """builds learning to rank layer
        """
        if len(self.out_layers) > 0:
            res = [torch.nn.Linear(self.kernel_num, self.out_layers[0]),
                   torch.nn.ReLU()
                  ]
            for i in range(1, len(self.out_layers)):
                res += [torch.nn.Linear(self.out_layers[i-1], self.out_layers[i]),
                        torch.nn.ReLU()
                       ]
            res += [torch.nn.Linear(self.out_layers[-1], 1)]
        else:
            res = [torch.nn.Linear(self.kernel_num, 1)]
    
        res = torch.nn.Sequential(*res)
        return res
        
    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        """forward step
        """
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)
        logits_diff = logits_1 - logits_2
        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        """Translation layer
        """
        def sim_matrix(a, b, eps=1e-8):
            a_n, b_n = a.norm(dim=2), b.norm(dim=2)
            a_n = torch.unsqueeze(a_n, dim=1) + eps
            b_n = torch.unsqueeze(b_n, dim=1).transpose(2, 1) + eps
            sim_mt = torch.matmul(b, a.transpose(2, 1))
            sim_mt = (sim_mt/a_n)/b_n
            return sim_mt
        
        doc_emb = self.embeddings(doc)
        query_emb = self.embeddings(query)
        return sim_matrix(doc_emb, query_emb)

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        """Kernel pooling
        """
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        """brings all together
        """
        # shape = [Batch, Left], [Batch, Right]
        query, doc = inputs['query'], inputs['document']
        
        # shape = [Batch, Left, Right]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape = [Batch, Kernels]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape = [Batch]
        out = self.mlp(kernels_out)
        return out