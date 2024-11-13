import torch
import torch.nn as nn


class CustomWeightedLoss(nn.Module):
    def __init__(self, weight_default, weight_low, weight_mid, weight_high, weight_highest):
        super(CustomWeightedLoss, self).__init__()
        self.weight_default = weight_default
        self.weight_low = weight_low
        self.weight_mid = weight_mid
        self.weight_high = weight_high
        self.weight_highest = weight_highest

    def forward(self, predictions, targets):
        # 计算均方误差
        mse = torch.mean((predictions - targets) ** 2)

        # 根据真实标签的范围分配权重
        weights = torch.where((targets == 1) | (targets == 9), self.weight_highest,
                              torch.where((targets == 2) | (targets == 8), self.weight_high,
                                          torch.where((targets == 3) | (targets == 4) | (targets == 7), self.weight_mid,
                                                      torch.where((targets == 5) | (targets == 6), self.weight_low,
                                                                  self.weight_default))))

        # 计算加权损失
        weighted_loss = mse * weights

        # 计算平均损失
        loss = torch.mean(weighted_loss)

        return loss
