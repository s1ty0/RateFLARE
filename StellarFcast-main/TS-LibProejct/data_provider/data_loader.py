import os

import numpy as np
import torch
from torch.utils.data import Dataset

# 可选：仅在需要文本编码时导入（避免无用依赖）
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class FluxDataLoader(Dataset):
    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None, on_enhance=False, encoder="minLM", on_mm_statistics=False, on_mm_history=False):
        self.args = args
        self.flag = flag
        self.encoder = encoder
        # 是否开启多模态
        self.on_mm_statistics = on_mm_statistics
        self.on_mm_history = on_mm_history
        self.on_enhance = on_enhance # 是否开启单模态数据增强（引入差分）

        # === 文本编码器路径映射（可扩展）===
        ENCODER_PATH_MAP = {
            "minLM": "./textEncoder/all-MiniLM-L6-v2",
            "mpnet": "./textEncoder/all-mpnet-base-v2",
            "bert-chinese": "textEncoder/bert-base-chinese"
            # 未来可加： "bge": "./textEncoder/bge-small-en-v1.5", ...
        }

        if self.encoder not in ENCODER_PATH_MAP:
            raise ValueError(f"Unsupported encoder: {self.encoder}. Choose from {list(ENCODER_PATH_MAP.keys())}")

        self.text_encoder_path = ENCODER_PATH_MAP[self.encoder]

        # 确定数据路径
        if flag == 'TRAIN':
            data_dir = f"{root_path}/train"
        elif flag == 'TEST':
            data_dir = f"{root_path}/test"
        elif flag == 'VAL':
            data_dir = f"{root_path}/val"
        else:
            data_dir = root_path

        # 加载数据
        lc_data = np.load(f"{data_dir}/lc_data.npy")      # (N, 512)
        label_data = np.load(f"{data_dir}/label_data.npy")  # (N,)

        # 本地代码一律执行小样本，模式，
        # lc_data = lc_data[0:30]
        # label_data = label_data[0:30]

        # 应用 limit_size，暂且保留，和以上的语句一个意思。
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:
                limit_size = int(limit_size * len(lc_data))
            lc_data = lc_data[:limit_size]
            label_data = label_data[:limit_size]

        self.X = lc_data      # (N, 512)
        self.y = label_data   # (N,)
        print(f"[{flag}] Loaded {len(self.X)} samples.")

        # # === Debug: 打印 peak_ratio 分布 ===
        # flare_mask = (self.y == 1)
        # no_flare_mask = (self.y == 0)
        # pr_f = [np.max(x) / (np.median(x) + 1e-8) for x in self.X[flare_mask]]
        # pr_nf = [np.max(x) / (np.median(x) + 1e-8) for x in self.X[no_flare_mask]]
        # print("Flare: median PR={:.2f}, min={:.2f}".format(np.median(pr_f), np.min(pr_f)))
        # print("No-flare: median PR={:.2f}, max={:.2f}".format(np.median(pr_nf), np.max(pr_nf)))


        # === 预计算全局统计量（用于 idea3 / idea4）===
        # print(f"[{flag}] 预计算全局统计量...")
        # self.global_stats = []
        # for x in self.X:
        #     x_med = np.median(x)
        #     x_max = np.max(x)
        #     t_max = np.argmax(x)
        #     peak_ratio = x_max / (x_med + 1e-8)
        #     rise_rate = (x_max - x[0]) / max(t_max, 1)
        #     half_max = x_med + (x_max - x_med) * 0.5
        #     dur = np.sum(x > half_max) if np.any(x > half_max) else 0
        #     self.global_stats.append({
        #         'peak_ratio': peak_ratio,
        #         'rise_rate': rise_rate,
        #         'duration': dur,
        #         'x_max': x_max,
        #         'x_med': x_med
        #     })

        # === 预计算文本嵌入 ===
        self.text_embeddings = None
        self.history_embeddings = None

        # ============== 修改后  =================
        if self.on_mm_statistics:
            emb_file = os.path.join(data_dir, f"text_embeddings_{encoder}.npy")
            if os.path.exists(emb_file):
                self.text_embeddings = np.load(emb_file)
                print(f"[{flag}] Loaded text embeddings from {emb_file}, shape: {self.text_embeddings.shape}")
            else:
                raise FileNotFoundError(
                    f"Text embeddings not found at {emb_file}. Please run generate_text_embeddings.py first.")

        if self.on_mm_history: #  f"text_embeddings_his_red_{args.encoder}.npy"
            emb_file = os.path.join(data_dir, f"text_embeddings_his_red_{encoder}.npy")
            if os.path.exists(emb_file):
                self.history_embeddings = np.load(emb_file)
                print(f"[{flag}] Loaded (history) text embeddings from {emb_file}, shape: {self.history_embeddings.shape}")
            else:
                raise FileNotFoundError(
                    f"Text embeddings not found at {emb_file}. Please run generate_history_embeddings.py first.")


    def _get_activity_level(self, stats):
        x_max = stats['x_max']
        x_med = stats['x_med']
        peak_ratio = stats['peak_ratio']
        duration = stats['duration']

        # 1. 处理极端异常值（no-flare 中出现的超高 PR）
        if peak_ratio > 100.0:
            return "无显著活动"  # 极可能是噪声/异常，非真实耀斑

        # 2. x_max 过低 → 无活动（但阈值降低，因归一化后值偏小）
        if x_max < 0.1:  # 原为 0.2，现在更敏感
            return "无显著活动"

        # 3. 【关键修改】不再用 PR < 1.5 判为无活动！
        #    因为 flare 的 PR 可低至 1.0
        #    改为：只有当 PR 极低 + x_max 低 时才判为无活动
        if peak_ratio < 1.2 and x_max < 0.15:
            return "无显著活动"

        # 4. 微弱活动（覆盖 PR 1.2 ~ 2.5）
        if peak_ratio < 2.5:
            return "微弱增强行为"

        # 5. 明显活动（PR >= 2.5）
        if peak_ratio >= 2.5:
            # 至少持续 2 个时间步（原为 3，可适当放宽）
            if duration < 2:
                return "微弱增强行为"  # 瞬时尖峰，可能噪声

            # 强耀斑：高 PR + 高幅度 + 持续
            if peak_ratio >= 6.0 and x_max >= 0.5:
                return "强烈的爆发行为"
            else:
                return "明显的瞬态增强行为"

        # 兜底
        return "微弱增强行为"

    def _build_prompt(self, stats):
        activity = self._get_activity_level(stats)
        # 处理 peak_ratio（已是标量，直接用）
        peak_ratio = stats['peak_ratio']
        # 对 rise_rate 数组取均值（转换为标量）
        rise_rate_mean = stats['rise_rate'].mean()
        # duration 已是标量，直接用
        duration = stats['duration']

        return (
            f"在本次观测周期内，耀斑活动的全局统计特征如下："
            f"峰值通量相对于背景中位数的倍数为{peak_ratio:.2f}，"
            f"从起始到峰值的上升速率均值为{rise_rate_mean:.4f}单位/时间步，"
            f"爆发持续时间（半高全宽）为{duration:.1f}个时间步。"
            f"该序列表现出{activity}。"
        )

    def _enhance_flux(self, x):
        """
        仅保留原始信号和一阶差分（2通道）
        x: (512,) 或 (1, 512) → 统一处理为 (512,)
        Returns: (512, 2)
        """
        # 标准化输入为 (512,)
        if x.ndim == 2:
            if x.shape[0] == 1:
                x = x.squeeze(0)  # (1, 512) -> (512,)
            elif x.shape[1] == 512:
                x = x[0]  # 保守处理
            else:
                raise ValueError(f"Unexpected x shape: {x.shape}")
        elif x.ndim != 1:
            raise ValueError(f"Invalid x ndim: {x.ndim}")

        # 1. 原始信号
        feat1 = x  # (512,)

        # 2. 一阶差分
        feat2 = np.zeros_like(x)
        feat2[1:] = np.diff(x)

        # 拼接为 (512, 2)
        features = np.stack([feat1, feat2], axis=1)

        return features

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_raw = self.X[idx]  # (1，512)
        y = self.y[idx]

        # 确保x_raw变成 (512,)
        if x_raw.ndim == 2 and x_raw.shape[0] == 1:
            x_raw = x_raw.squeeze(0)  # (1, 512) → (512,)
        elif x_raw.ndim == 2 and x_raw.shape[1] == 1:
            x_raw = x_raw.squeeze(1)  # (512, 1) → (512,)

        # 保留原始光变曲线, 物理损失函数需要用到
        raw_lc = x_raw.copy()  # (512,) —— 注意：确保是 numpy array 或可转 tensor

        # 扩充一个维度
        x_final = x_raw[:, None]  # (512, 1)

        # 3个改进点都开启 [mm, mm_history, enhance]
        if self.on_mm_statistics and self.on_enhance and self.on_mm_history:
            x_final = self._enhance_flux(x_raw)  # (512, 2)
            text_emb = self.text_embeddings[idx].astype(np.float32)  # (384,)
            his_emb = self.history_embeddings[idx].astype(np.float32)  # (384,)
            return {"x_enc": x_final, "text_emb": text_emb, "his_emb": his_emb, "raw_lc": raw_lc}, torch.tensor(y,
                                                                                                                dtype=torch.long)

        # 开启两个 [mm, enhance]
        if self.on_mm_statistics and self.on_enhance:
            x_final = self._enhance_flux(x_raw)  # (512, 2)
            text_emb = self.text_embeddings[idx].astype(np.float32)  # (384,)
            return {"x_enc": x_final, "text_emb": text_emb, "his_emb": None, "raw_lc": raw_lc}, torch.tensor(y,
                                                                                                             dtype=torch.long)

        # 开启两个 [mm_his, enhance]
        if self.on_mm_history and self.on_enhance:
            x_final = self._enhance_flux(x_raw)  # (512, 2)
            his_emb = self.history_embeddings[idx].astype(np.float32)  # (384,)
            return {"x_enc": x_final, "text_emb": None, "his_emb": his_emb, "raw_lc": raw_lc}, torch.tensor(y,
                                                                                                            dtype=torch.long)

        # 开启两个 [mm, mm_his]
        if self.on_mm_history and self.on_mm_statistics:
            text_emb = self.text_embeddings[idx].astype(np.float32)  # (384,)
            his_emb = self.history_embeddings[idx].astype(np.float32)  # (384,)
            return {"x_enc": x_final, "text_emb": text_emb, "his_emb": his_emb, "raw_lc": raw_lc}, torch.tensor(y,
                                                                                                                dtype=torch.long)

        # 开启一个 [enhance]
        if self.on_enhance:
            x_final = self._enhance_flux(x_raw)  # (512, 2)
            return {"x_enc": x_final, "text_emb": None, "his_emb": None, "raw_lc": raw_lc}, torch.tensor(y,
                                                                                                         dtype=torch.long)

        # 开启一个 [mm]
        if self.on_mm_statistics:
            # 仅文本嵌入（广播）
            text_emb = self.text_embeddings[idx].astype(np.float32)  # (384,)
            return {"x_enc": x_final, "text_emb": text_emb, "his_emb": None, "raw_lc": raw_lc}, torch.tensor(y,
                                                                                                             dtype=torch.long)

        # 开启一个 [mm_his]
        if self.on_mm_history:
            # 仅文本嵌入（广播）
            his_emb = self.history_embeddings[idx].astype(np.float32)  # (384,)
            return {"x_enc": x_final, "text_emb": None, "his_emb": his_emb, "raw_lc": raw_lc}, torch.tensor(y,
                                                                                                            dtype=torch.long)

        return {"x_enc": x_final, "text_emb": None, "his_emb": None, "raw_lc": x_final}, torch.tensor(y,
                                                                                                      dtype=torch.long)
