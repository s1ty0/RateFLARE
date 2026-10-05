# -*- coding: utf-8 -*-
"""
Phase 1.3 文本嵌入导出 — 在统一协议数据上复刻 StellarF 的 FSI/FHR 文本模块
=========================================================================
复刻逻辑 (与 DataProcessProject/generate_statistics_embeddings.py / generate_history_embeddings.py 一致):
  FSI (text_embeddings_{encoder}.npy): prompt = "在本次观测周期内，耀斑持续了{N}个时间步，流量中值为{M}。"
    N = 输入窗口内耀斑活跃时间步数 (wlabel.npy 行和), M = 窗口通量中位数 (Xhat)
  FHR (text_embeddings_his_red_{encoder}.npy): prompt = 输入窗口内耀斑时间点分布描述
    (其原版用 "第 a、b、c... 时间步发生耀斑" 的模板; 超过 20 个取前 20)
编码器: bert-chinese (与 StellarF LLMProejct 默认 roberta-c 配套) 或 minLM。
产物写入 export_tslib_layout.py 生成的布局目录 ({out}/{split}/), LLMProejct --all 直接可用。

用法 (在 LLMProejct 的 Python 环境, 需装 sentence-transformers 且 textEncoder 已本地化):
    python ../export_embeddings_layout.py --layout <baselines/kepler_h480> --encoder bert-chinese \
        --text_encoder_dir ./textEncoder/bert-base-chinese
"""
import argparse
import os

import numpy as np


def build_fsi_prompts(wlabel, xhat, lang="zh"):
    counts = wlabel.sum(axis=1).astype(int)
    meds = np.median(xhat, axis=1)
    prompts = []
    for c, m in zip(counts, meds):
        if lang == "en":
            prompts.append(f"During this observation period, flares lasted {int(c)} time steps, "
                           f"with a median flux of {m:.4f}.")
        else:
            if c == 0:
                prompts.append(f"在本次观测周期内，耀斑持续了0个时间步，流量中值为{m:.4f}。")
            else:
                prompts.append(f"在本次观测周期内，耀斑持续了{int(c)}个时间步，流量中值为{m:.4f}。")
    return prompts


def build_fhr_prompts(wlabel, lang="zh"):
    prompts = []
    for row in wlabel:
        times = np.argwhere(row == 1).flatten().tolist()
        if lang == "en":
            if not times:
                prompts.append("No flare events were detected during this observation period.")
            else:
                head = times[:20]
                t_str = ", ".join(f"step {t}" for t in head)
                tail = "" if len(times) <= 20 else f" ({len(times)} in total, first 20 listed)"
                prompts.append(f"During this observation period, flares occurred at: {t_str}{tail}.")
        else:
            if not times:
                prompts.append("在本次观测周期内，未检测到任何耀斑事件。")
            else:
                head = times[:20]
                t_str = "、".join(f"第{t}时间步" for t in head)
                tail = "" if len(times) <= 20 else f"（共{len(times)}次，此处仅列前20次）"
                prompts.append(f"在本次观测周期内，耀斑分别发生在{t_str}{tail}。")
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True, help="export_tslib_layout.py 生成的目录")
    ap.add_argument("--lang", default="zh", choices=["zh", "en"], help="prompt 语言 (2.7 对照实验)")
    ap.add_argument("--encoder", default="bert-chinese",
                    help="仅用于文件命名 (其 loader 按此拼文件名); 实际编码器由 --text_encoder_dir 决定")
    ap.add_argument("--save_as", default=None,
                    help="文件名里的 encoder 槽位 (默认=--encoder; 英文嵌入想复用原槽位名可传 bert-chinese)")
    ap.add_argument("--text_encoder_dir", default="./textEncoder/bert-base-chinese")
    a = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(a.text_encoder_dir)
    slot = a.save_as or a.encoder

    for split in ["train", "val", "test"]:
        d = os.path.join(a.layout, split)
        wlabel = np.load(os.path.join(d, "wlabel.npy"))
        xhat = np.load(os.path.join(d, "lc_data.npy"))  # 已 min-max 归一化 (中位数口径一致即可)
        fsi = enc.encode(build_fsi_prompts(wlabel, xhat, a.lang), batch_size=64,
                         show_progress_bar=True, normalize_embeddings=False)
        fhr = enc.encode(build_fhr_prompts(wlabel, a.lang), batch_size=64,
                         show_progress_bar=True, normalize_embeddings=False)
        np.save(os.path.join(d, f"text_embeddings_{slot}.npy"), fsi.astype(np.float32))
        np.save(os.path.join(d, f"text_embeddings_his_red_{slot}.npy"), fhr.astype(np.float32))
        print(f"[{split}] lang={a.lang} enc={a.text_encoder_dir} → 存为 *_{slot}.npy ({fsi.shape}/{fhr.shape})")
    print("完成。2.7 四组对照: (1) zh+bert-chinese (2) en+bert-base-uncased "
          "(3) en+multilingual-e5 (4) 数值直连=不加载嵌入文件即可。")


if __name__ == "__main__":
    main()
