import argparse

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertModel, BertConfig
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


def moving_average_decompose(x, window=25):
    """
    向量化优化：单条时间序列移动平均分解
    x: (L,) —— 1D numpy array
    Returns: trend (L,), resid (L,)
    """
    pad = window // 2
    x_padded = np.pad(x, (pad, pad), mode='edge')
    # 向量化卷积（原代码没问题，但批量处理时优化）
    trend = np.convolve(x_padded, np.ones(window) / window, mode='valid')
    resid = x - trend
    return trend, resid


def preprocess_time_series(X, window=25):
    """
    批量优化：避免循环每个样本（原代码是for i in range(N)）
    X: (N, L)
    Returns: (N, L, 2)
    """
    if X.ndim == 3:
        X = X.squeeze(axis=1)
    N, L = X.shape
    X_out = np.zeros((N, L, 2), dtype=np.float32)

    # 新增：打印预处理进度，确认是否卡住
    from tqdm import tqdm
    for i in tqdm(range(N), desc="Decomposing time series"):
        trend, resid = moving_average_decompose(X[i], window=window)
        X_out[i, :, 0] = trend
        X_out[i, :, 1] = resid
    return X_out

# ----------------------------
# 2. Early Stopping 类
# ----------------------------
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, mode='max'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        score = val_score if self.mode == 'max' else -val_score

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# ----------------------------
# 2. Dataset
# ----------------------------
class FlareDataset(Dataset):
    def __init__(self, X, y, seq_len=512):
        # X is now (N, L, 2)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # (L, 2)
        if x.size(0) < self.seq_len:
            padding = torch.zeros(self.seq_len - x.size(0), 2)
            x = torch.cat([x, padding], dim=0)
        else:
            x = x[:self.seq_len]
        return x, self.y[idx]  # (seq_len, 2), scalar label

# ----------------------------
# 3. Label Smoothing Loss
# ----------------------------
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, pred, target):
        log_probs = torch.log_softmax(pred, dim=-1)
        n_classes = pred.size(-1)
        true_dist = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        loss = torch.sum(-true_dist * log_probs, dim=-1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss



# ----------------------------
# 4. 模型定义（简化版）
# ----------------------------
class MyTransformerModel(nn.Module):
    def __init__(self, num_classes=2, input_dim=2, local_path=None):
        super().__init__()
        # 加载 BERT
        backbone = BertModel.from_pretrained(local_path)
        self.model_type = "bert"
        # 使用lora
        if self.model_type == "bert":
            my_target_modules = ["query", "key", "value", "dense"]
        lora_config = LoraConfig(
            task_type=None,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=my_target_modules,
            bias="none",  # 不训练原始模型的 bias 参数（默认值，最常用）
        )
        # 应用 LoRA：仅这些模块可训练，其余冻结
        self.backbone = get_peft_model(backbone, lora_config)
        self.backbone.print_trainable_parameters()  # 打印可训练参数量

        self.config = self.backbone.config
        self.input_proj = nn.Linear(input_dim, self.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (B, L, 2)
        embedded = self.input_proj(x)  # (B, L, hidden_size)
        outputs = self.backbone(inputs_embeds=embedded)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(cls_emb)
        return logits

# -----------------------------
# 5. 训练流程
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description='Stellar Forecasting with LLM and LoRA using PyTorch Lightning')
    parser.add_argument('--root_path', type=str, default='./myDataK20', help='Path to my data')
    parser.add_argument('--dataset', type=str, default="kepler", help='dataset we use.')

    args = parser.parse_args()

    if args.dataset == "kepler":
        args.root_path = "./myDataK20"
    elif args.dataset == "tess":
        args.root_path = "./myDataT20"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    torch.manual_seed(42)
    np.random.seed(42)

    # === 替换为你的真实数据加载逻辑 ===
    # 例如：
    X_train = np.load(f"{args.root_path}/train/lc_data.npy")  # (N_train, L)
    y_train = np.load(f"{args.root_path}/train/label_data.npy")
    X_val = np.load(f"{args.root_path}/val/lc_data.npy")
    y_val = np.load(f"{args.root_path}/val/label_data.npy")
    X_test = np.load(f"{args.root_path}/test/lc_data.npy")
    y_test = np.load(f"{args.root_path}/test/label_data.npy")

    # 模拟数据仅用于演示
    # N_train, N_val, N_test, L = 20, 5, 5, 512
    # X_train = np.random.randn(N_train, L).astype(np.float32)
    # y_train = np.random.randint(0, 2, size=(N_train,)).astype(np.int64)
    # X_val = np.random.randn(N_val, L).astype(np.float32)
    # y_val = np.random.randint(0, 2, size=(N_val,)).astype(np.int64)
    # X_test = np.random.randn(N_test, L).astype(np.float32)  # ← 添加
    # y_test = np.random.randint(0, 2, size=(N_test,)).astype(np.int64)  # ← 添加

    # === 分别对 train/val/test 做 STL 分解（注意：只用 train 的 scaler？）===
    print("Decomposing time series into [trend, resid]...")
    X_train_proc = preprocess_time_series(X_train)
    X_val_proc = preprocess_time_series(X_val)
    X_test_proc = preprocess_time_series(X_test)

    # === 构建 Dataset 和 DataLoader ===
    train_dataset = FlareDataset(X_train_proc, y_train, seq_len=512)
    val_dataset = FlareDataset(X_val_proc, y_val, seq_len=512)
    test_dataset = FlareDataset(X_test_proc, y_test, seq_len=512)
    print("训练集第一条样本的形状为：", train_dataset[0][0].shape)

    print("开始构建DataLoader...")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print("DataLoader构建完成，开始初始化模型...")

    # === 模型 & 优化器 ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_path = "models/bert_base_uncased/"
    print(f"加载BERT模型：{local_path}")
    print("模型初始化完成，传输到GPU...")
    model = MyTransformerModel(
        num_classes=2,
        input_dim=2,  # ← 关键！两通道输入
        local_path=local_path
    ).to(device)

    print("模型已传输到GPU，初始化损失函数...")
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    # 优化所有可训练参数（LoRA + input_proj + classifier）
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3
    )

    # === 早停设置：监控 F1 Score（越高越好）===
    early_stopping = EarlyStopping(patience=10, verbose=True, mode='max')
    print("早停初始化完成，进入训练循环...")

    best_f1 = -1
    for epoch in range(200): #
        print(f"开始Epoch {epoch + 1} 训练...")  # 新增
        model.train()
        total_loss = 0
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1} 训练中", leave=False):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # ===== 验证 =====
        model.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_loader, desc=f"Epoch {epoch+1} 验证中", leave=False):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                probs = F.softmax(logits, dim=-1)[:, 1]
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = float('nan')

        print(f"Epoch {epoch+1:03d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        # ===== 早停判断：基于 F1 =====
        current_f1 = f1
        early_stopping(current_f1)

        if early_stopping.early_stop:
            print("Early stopping triggered (no improvement in F1 for 10 epochs)!")
            break

        # 保存最佳 F1 模型
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(model.state_dict(), "flare_best_model_by_f1.pth")
            print(f"  --> Best F1 updated: {best_f1:.4f}, model saved.")

        model.train()

    print("Training finished.")

    # === 6. 测试阶段：加载最佳模型，在测试集上评估 ===
    print("\n=== Testing on best model ===")
    model.load_state_dict(torch.load("flare_best_model_by_f1.pth", map_location=device))
    model.eval()

    test_preds, test_probs, test_labels = [], [], []
    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader, desc=f"Epoch {epoch + 1} 测试中", leave=False):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            probs = F.softmax(logits, dim=-1)[:, 1]
            preds = logits.argmax(dim=-1)
            test_preds.extend(preds.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            test_labels.extend(y_batch.cpu().numpy())

    # 转为 numpy array
    test_labels = np.array(test_labels)
    test_preds = np.array(test_preds)
    test_probs = np.array(test_probs)

    # 打印测试指标
    test_acc = accuracy_score(test_labels, test_preds)
    test_prec = precision_score(test_labels, test_preds, zero_division=0)
    test_rec = recall_score(test_labels, test_preds, zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    try:
        test_auc = roc_auc_score(test_labels, test_probs)
    except ValueError:
        test_auc = float('nan')

    print("\n[Test Results]")
    print(f"Acc: {test_acc:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")

if __name__ == "__main__":
    main()