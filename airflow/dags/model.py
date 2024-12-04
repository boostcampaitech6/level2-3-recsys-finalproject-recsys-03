import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        # 모델 구조 정의

    def forward(self, x):
        # 데이터를 모델에 통과시키는 방법 정의
        return x