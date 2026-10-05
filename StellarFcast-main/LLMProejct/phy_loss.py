import torch
import torch.nn as nn
import torch.nn.functional as F



# 使用实例：
# class MyTransformerLightningModule(LightningModule):
#     def __init__(self, ..., lambda_phys=0.1):
#         super().__init__()
#         self.criterion = PhysicsRegularizedLoss(lambda_phys=lambda_phys)
#
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = self.criterion(logits, y, x)  # 传入原始输入 x
#         ...



