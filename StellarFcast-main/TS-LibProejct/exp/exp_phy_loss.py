import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsRegularizedLoss(nn.Module): # kepler 0.017093 ✅  tess: 0.013866
    def __init__(self, lambda_phys=0.1, rise_threshold=0.017093, conf_threshold = 0.5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_phys = lambda_phys
        self.rise_threshold = rise_threshold
        self.conf_threshold = conf_threshold  # ← 新增参数, 可调的置信度阈值, 默认平衡点，为了稳定物理正则化约束

    def forward(self, logits, targets, input_lc):
        # 原始分类损失
        ce = self.ce_loss(logits, targets)

        # 物理约束损失：仅对真实耀斑（label=1）的样本进行计算
        flare_mask = (targets == 1).float() # [B] 返回值是一个布尔数组吗？
        if flare_mask.sum() == 0:
            phys_loss = torch.tensor(0.0, device=logits.device)
        else:
            phys_loss = self.flare_shape_penalty_on_true_flare(input_lc, flare_mask)

        return ce + self.lambda_phys * phys_loss

    def flare_shape_penalty_on_true_flare(self, input_lc, flare_mask):
        diff = input_lc[:, 1:] - input_lc[:, :-1]  # [B, L-1]

        max_rise = torch.max(diff, dim=1).values  # [B]# 对真实耀斑：若 max_rise < threshold， 则惩罚
        penalty = torch.relu(self.rise_threshold - max_rise)  # [B]#只对真实耀斑样本计算损失
        weighted_penalty = penalty * flare_mask

        return weighted_penalty.sum() / (flare_mask.sum() + 1e-8)


    def flare_shape_penalty(self, input_lc, pred_probs):
        """
        lc: [B, L] 原始光变曲线
        pred_prob: [B] 模型预测为耀斑的概率
        返回：违反耀斑形状先验的惩罚
        """
        # 计算一阶导数（近似上升/下降速率）
        diff = input_lc[:, 1:] - input_lc[:, :-1]  # [B, L-1]

        # 耀斑应有显著上升段
        max_rise = torch.max(diff, dim=1).values  # [B]

        # 若预测是耀斑但无显著上升，则惩罚
        penalty = torch.relu(self.rise_threshold - max_rise)

        # 加权: 只惩罚高置信度预测（pred_prob > 0.5）# pred_prob 参数可调
        weight = torch.clamp(pred_probs - self.conf_threshold, min=0.0)

        return ((penalty * weight).mean() * weight).mean()

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



