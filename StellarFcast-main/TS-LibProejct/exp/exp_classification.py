from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from exp.exp_phy_loss import (
    PhysicsRegularizedLoss,
)  # 此处的已经是第二版，实验效果更好的phy_loss了

# from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
# 将EarlyStopping修改为F2分数 -Tu
from utils.tools import adjust_learning_rate, cal_accuracy
from utils.myTools import EarlyStopping

import torch
import torch.nn as nn
import torch_optimizer as optim
import os
import time
from datetime import datetime, timezone, timedelta
import warnings
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    fbeta_score,
    average_precision_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _prepare_batch(self, batch):  # 编写预解包逻辑
        inputs, label = batch
        x_enc = inputs["x_enc"].float().to(self.device)
        text_emb = (
            inputs["text_emb"].float().to(self.device)
            if inputs["text_emb"] is not None
            else None
        )
        his_emb = (
            inputs["his_emb"].float().to(self.device)
            if inputs["his_emb"] is not None
            else None
        )
        raw_lc = inputs["raw_lc"].float().to(self.device)
        return x_enc, text_emb, his_emb, label.to(self.device), raw_lc

    def _build_model(self):
        self.args.seq_len = 512  # 这里的值需要不断更改来调参
        self.args.pred_len = 0  # 这里是0感觉不对，调试一下 ,分类任务中用不到，正确
        self.args.num_class = 2

        # 3. 加载模型
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()  #
        if self.args.on_phy_loss:
            criterion = PhysicsRegularizedLoss()
        return criterion

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="TEST")
        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"))
            )

        self.model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for batch in test_loader:
                x_enc, text_emb, his_emb, label, _ = self._prepare_batch(batch)
                outputs = self.model(
                    x_enc, None, None, text_emb=text_emb, his_emb=his_emb
                )
                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.softmax(preds, dim=1)
        probs_positive = probs[:, 1].cpu().numpy()
        trues = trues.flatten().cpu().numpy()

        import os
        from sklearn.metrics import matthews_corrcoef, brier_score_loss

        _ds = "kepler" if "kepler" in str(self.args.root_path) else "tess"
        _h = "480" if "h480" in str(self.args.root_path) else "48"

        def _ece(y, p, bins=15):
            if p.min() == p.max():
                return 1.0
            edges = np.linspace(0, 1, bins + 1)
            return float(
                sum(
                    ((p > edges[i]) & (p <= edges[i + 1])).sum()
                    / len(y)
                    * abs(
                        p[(p > edges[i]) & (p <= edges[i + 1])].mean()
                        - y[(p > edges[i]) & (p <= edges[i + 1])].mean()
                    )
                    for i in range(bins)
                    if ((p > edges[i]) & (p <= edges[i + 1])).sum()
                )
            )

        def _rec_at_fpr(y, p, fpr=0.1):
            order = np.argsort(-p)
            ys = y[order]
            n_pos, n_neg = y.sum(), (1 - y).sum()
            if n_pos == 0 or n_neg == 0:
                return float("nan")
            tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
            hits = np.argwhere(fp / n_neg <= fpr).flatten()
            return float(tp[hits[-1]] / n_pos) if len(hits) else 0.0

        def _block(y, p, thr):
            pred = (p >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            n = tn + fp + fn + tp
            e_corr = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / n if n else 0.0
            return {
                "thr": thr,
                "rec": recall_score(y, pred, pos_label=1, zero_division=0),
                "prec": precision_score(y, pred, pos_label=1, zero_division=0),
                "f1": f1_score(y, pred, pos_label=1, zero_division=0),
                "f2": fbeta_score(y, pred, beta=2.0, pos_label=1, zero_division=0),
                "mcc": float(matthews_corrcoef(y, pred)),
                "csi": tp / (tp + fp + fn) if (tp + fp + fn) else float("nan"),
                "hss": (
                    ((tp + tn) - e_corr) / (n - e_corr)
                    if n and n != e_corr
                    else float("nan")
                ),
                "freq_bias": (tp + fp) / (tp + fn) if (tp + fn) else float("nan"),
                "acc": accuracy_score(y, pred),
                "cm": (tn, fp, fn, tp),
            }

        # --- 与阈值无关的主指标 ---
        auc_roc = (
            roc_auc_score(trues, probs_positive)
            if len(np.unique(trues)) > 1
            else float("nan")
        )
        auc_pr = (
            average_precision_score(trues, probs_positive)
            if len(np.unique(trues)) > 1
            else float("nan")
        )
        rec_fpr01 = _rec_at_fpr(trues, probs_positive)
        brier = float(brier_score_loss(trues, probs_positive))
        ece15 = _ece(trues, probs_positive)

        # --- 验证集寻最优 F1 阈值 (协议口径N) ---
        best_thr = 0.5
        vp_path = f"../preds/{_ds}/{setting}/val_prob{_h}.npy"
        vt_path = f"../preds/{_ds}/{setting}/val_true{_h}.npy"
        if os.path.exists(vp_path) and os.path.exists(vt_path):
            vp = np.load(vp_path)
            vt = np.load(vt_path).astype(int)
            if len(np.unique(vp)) > 1:
                best_f1, best_thr = -1, 0.5
                for t in np.unique(vp):
                    f1t = f1_score(
                        vt, (vp >= t).astype(int), pos_label=1, zero_division=0
                    )
                    if f1t > best_f1:
                        best_f1, best_thr = f1t, float(t)
        else:
            print(
                "[提示] 未找到 val_prob/val_true → 最优阈值回退 0.5 (vali 需保存 val_true)"
            )

        b05 = _block(trues, probs_positive, 0.5)  # 口径B/可比参考
        bop = _block(trues, probs_positive, best_thr)  # 口径N 运行点

        # --- 导出 (供 eval_predictions 统一评估) ---
        os.makedirs(f"../preds/{_ds}/{setting}", exist_ok=True)
        np.save(f"../preds/{_ds}/{setting}/test_prob{_h}.npy", probs_positive)
        np.save(f"../preds/{_ds}/{setting}/test_true{_h}.npy", trues.astype(np.int64))

        # --- 打印 ---
        print("\n" + "=" * 56)
        print(f"【测试集结果】{setting}")
        print("=" * 56)
        print(f"★ AUPRC (主指标): {auc_pr:.6f}   AUROC: {auc_roc:.6f}")
        print(
            f"  Brier: {brier:.6f}   ECE15: {ece15:.6f}   Recall@FPR=0.1: {rec_fpr01:.6f}"
        )
        print(f"\n[运行点: 验证集最优F1阈值 t*={best_thr:.4f}]  ← 论文口径N")
        print(
            f"  Recall: {bop['rec']:.6f}  Precision: {bop['prec']:.6f}  F1: {bop['f1']:.6f}  F2: {bop['f2']:.6f}"
        )
        print(
            f"  FrequencyBias: {bop['freq_bias']:.6f}  MCC: {bop['mcc']:.6f}  CSI: {bop['csi']:.6f}  HSS: {bop['hss']:.6f}"
        )
        print(f"  混淆矩阵 [TN FP FN TP]: {bop['cm']}")
        print(f"\n[参考: 阈值0.5]  ← 仅与两篇论文可比")
        print(
            f"  Acc: {b05['acc']:.6f}  Recall: {b05['rec']:.6f}  Precision: {b05['prec']:.6f}  F1: {b05['f1']:.6f}"
        )
        print(f"  混淆矩阵 [TN FP FN TP]: {b05['cm']}")
        print("=" * 56)

        # --- 保存到 results/ ---
        result_dir = "./results/"
        os.makedirs(result_dir, exist_ok=True)
        date_str = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")
        short = setting.replace(f"{self.args.task_name}_", "", 1)
        short = short.rsplit(f"_{self.args.des}_", 1)[0]
        fname = f"{date_str}_{short}.txt"
        with open(os.path.join(result_dir, fname), "a", encoding="utf-8") as f:
            f.write(f"{setting}\n")
            f.write(f"AUPRC: {auc_pr:.6f}\nAUROC: {auc_roc:.6f}\n")
            f.write(
                f"Brier: {brier:.6f}\nECE15: {ece15:.6f}\nRecall@FPR0.1: {rec_fpr01:.6f}\n"
            )
            f.write(f"t*: {best_thr:.6f}\n")
            f.write(
                f"Rec@t*: {bop['rec']:.6f}\nPrec@t*: {bop['prec']:.6f}\nF1@t*: {bop['f1']:.6f}\nF2@t*: {bop['f2']:.6f}\n"
            )
            f.write(
                f"FreqBias@t*: {bop['freq_bias']:.6f}\nMCC@t*: {bop['mcc']:.6f}\nCSI@t*: {bop['csi']:.6f}\nHSS@t*: {bop['hss']:.6f}\n"
            )
            f.write(f"Acc@0.5(参考): {b05['acc']:.6f}\nF1@0.5(参考): {b05['f1']:.6f}\n")
            f.write("-" * 50 + "\n")

        return {
            "auprc": auc_pr,
            "auroc": auc_roc,
            "brier": brier,
            "ece15": ece15,
            "recall_at_fpr0.1": rec_fpr01,
            "best_thr": best_thr,
            "operating_point": bop,
            "at_0.5": b05,
        }

    def vali(self, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                x_enc, text_emb, his_emb, label, raw_lc = self._prepare_batch(batch)
                outputs = self.model(
                    x_enc, None, None, text_emb=text_emb, his_emb=his_emb
                )  # (B, num_classes)

                if self.args.on_phy_loss:
                    loss = criterion(outputs, label, raw_lc)
                else:
                    loss = criterion(outputs, label.squeeze().long())
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label.detach())

        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)

        probs = torch.softmax(preds, dim=1)[:, 1].cpu().numpy()  # 正类概率
        trues_np = trues.flatten().cpu().numpy()

        # 计算 AUPRC（早停指标）
        val_auprc = (
            average_precision_score(trues_np, probs)
            if len(np.unique(trues_np)) > 1
            else 0.0
        )

        # 保存验证概率（如果有 setting）
        if getattr(self, "setting", None):
            import os

            _ds = "kepler" if "kepler" in str(self.args.root_path) else "tess"
            _h = "480" if "h480" in str(self.args.root_path) else "48"
            os.makedirs(f"../preds/{_ds}/{self.setting}", exist_ok=True)
            np.save(f"../preds/{_ds}/{self.setting}/val_prob{_h}.npy", probs)
            np.save(
                f"../preds/{_ds}/{self.setting}/val_true{_h}.npy",
                trues_np.astype(np.int64),
            )

        # 计算二分类指标（正类 label=1）
        predictions = (probs >= 0.5).astype(int)
        precision = precision_score(
            trues_np, predictions, pos_label=1, average="binary", zero_division=0
        )
        recall = recall_score(
            trues_np, predictions, pos_label=1, average="binary", zero_division=0
        )
        f1 = f1_score(
            trues_np, predictions, pos_label=1, average="binary", zero_division=0
        )

        # 打印监控指标（包括 Precision, Recall, F1, AUPRC）
        print(
            f"Val Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUPRC: {val_auprc:.4f}"
        )

        self.model.train()

        # ⚠️ 早停指标仍然使用 val_auprc（返回值第二个是 AUPRC）
        return np.average(total_loss), val_auprc

    def train(self, setting):
        self.setting = setting

        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        # 注意：不再加载 test_loader 用于训练监控

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)

        # 使用 mode='max'，因为 F2 越大越好
        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
            path=os.path.join(path, "checkpoint.pth"),
            mode="max",
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()  # 假设是 nn.CrossEntropyLoss()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_time = time.time()
            train_loss = []

            for i, batch in enumerate(train_loader):
                model_optim.zero_grad()

                x_enc, text_emb, his_emb, label, raw_lc = self._prepare_batch(batch)
                outputs = self.model(
                    x_enc, None, None, text_emb=text_emb, his_emb=his_emb
                )  # (base): (30,1,512)

                if self.args.on_phy_loss:
                    loss = criterion(outputs, label, raw_lc)
                else:
                    loss = criterion(outputs, label.squeeze().long())
                train_loss.append(loss.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )  # 建议 1.0
                model_optim.step()

                # if (i + 1) % 10 == 0:
                #     iter_time = time.time() - time_now
                #     speed = iter_time / (i + 1)
                #     left_time = speed * (
                #         (self.args.train_epochs - epoch) * train_steps - i
                #     )
                #     print(
                #         f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}"
                #     )
                #     print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")

            train_loss = np.average(train_loss)
            vali_loss, val_auprc = self.vali(vali_loader, criterion)  # 只用验证集

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")
            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                f"Train Loss: {train_loss:.3f}, Vali Loss: {vali_loss:.3f}, Vali AUPRC: {val_auprc:.3f}"
            )

            # 早停基于 F2-score（越大越好）
            early_stopping(val_auprc, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # 加载最佳模型
        best_model_path = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path))
        self.vali(vali_loader, criterion)  # 用最佳模型重跑一次, 导出其 val 概率
        return self.model
