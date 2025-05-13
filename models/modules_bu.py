import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'{self.__class__.__name__}{self.shape}'

    def forward(self, input):
        batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)
        return out


class CopyConcat(nn.Module):
    def __init__(self, n, dim):
        super().__init__()
        # dimは、(C, H, W) に対する処理を想定
        self.n = n
        self.dim = dim

    def __repr__(self):
        return f'{self.__class__.__name__}({self.n}, {self.dim})'

    def forward(self, input):
        # inputにはバッチ(B, C, H, W)が入る。画像のテンソルはdim=1からはじまるからdimに1たしてる
        out = torch.cat([input for _ in range(self.n)], dim=self.dim+1)
        return out


class SplitMerge(nn.Module):
    def __init__(self, chunks, dim, sum=False):
        super().__init__()
        # dimは、(C, H, W) に対する処理を想定
        self.chunks = chunks
        self.dim = dim
        self.sum = sum

    def __repr__(self):
        return f'{self.__class__.__name__}({self.chunks}, {self.dim})'

    def forward(self, input):
        x = input.view(input.shape[0], self.chunks, -1)

        # inputにはバッチ(B, C, H, W)が入る。画像のテンソルはdim=1からはじまるからdimに1たしてる
        if self.sum:
            x = torch.sum(x, dim=self.dim+1)
        else:
            x = torch.mean(x, dim=self.dim+1)
        return x

