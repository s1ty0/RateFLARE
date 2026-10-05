import torch
from data_provider.data_loader import FluxDataLoader
from torch.utils.data import DataLoader


def collate_fn(data):
    """
    同时支持单模态（仅时序特征）和多模态（时序+文本特征）输入
    - 单模态：每个样本为 (features, label)，其中 features 是时序数据
    - 多模态：每个样本为 ({"x_enc": 时序数据, "text_emb": 文本嵌入}, label)
    - 若没有文本嵌入，text_emb_batch 返回 None
    """
    # 解包数据：区分单模态和多模态格式
    if isinstance(data[0][0], dict):
        # 多模态格式：(dict, label)
        inputs, labels = zip(*data)
        # 提取时序特征
        x_enc_list = [inp["x_enc"] for inp in inputs]
        # 提取文本嵌入（可能为None）
        text_emb_list = [inp.get("text_emb") for inp in inputs]
        his_emb_list = [inp.get("his_emb") for inp in inputs]

        # 从 dict 中提取 raw_lc
        raw_lc_list = [inp["raw_lc"] for inp in inputs]
    else:
        # 单模态格式：(features, label)，默认features为时序数据
        features, labels = zip(*data)
        x_enc_list = features
        text_emb_list = [None] * len(features)  # 单模态时文本嵌入全为None
        his_emb_list = [None] * len(features)  # 单模态时文本嵌入全为None

        raw_lc_list = features  # fallback: raw_lc 等于输入特征

    # 处理时序输入：堆叠为 (B, L, C)
    x_enc_batch = torch.stack([
        torch.as_tensor(x, dtype=torch.float32) for x in x_enc_list
    ], dim=0)

    # 堆叠 raw_lc: 应为 (B, L) —— 确保原始光变曲线是二维
    raw_lc_batch = torch.stack([
        torch.as_tensor(x, dtype=torch.float32) for x in raw_lc_list
    ], dim=0)

    # 处理文本嵌入(statistics)：全为None则返回None，否则堆叠为 (B, D)
    if all(emb is None for emb in text_emb_list):
        text_emb_batch = None
    else:
        # 过滤掉None（理论上不会出现部分有部分无的情况）
        text_emb_batch = torch.stack([
            torch.as_tensor(emb, dtype=torch.float32)
            for emb in text_emb_list if emb is not None
        ], dim=0)
        # 若存在None但不全为None（异常情况），补充警告
        if len(text_emb_batch) != len(text_emb_list):
            import warnings
            warnings.warn("部分样本文本嵌入为None，已自动过滤")

    # 处理文本嵌入(history)：全为None则返回None，否则堆叠为 (B, D)
    if all(emb is None for emb in his_emb_list):
        his_emb_batch = None
    else:
        # 过滤掉None（理论上不会出现部分有部分无的情况）
        his_emb_batch = torch.stack([
            torch.as_tensor(emb, dtype=torch.float32)
            for emb in his_emb_list if emb is not None
        ], dim=0)
        # 若存在None但不全为None（异常情况），补充警告
        if len(his_emb_batch) != len(his_emb_list):
            import warnings
            warnings.warn("部分样本文本嵌入为None，已自动过滤")

    # 处理标签：堆叠为 (B, num_label)
    y_batch = torch.stack(labels, dim=0)

    return {"x_enc": x_enc_batch, "text_emb": text_emb_batch, "his_emb": his_emb_batch, "raw_lc": raw_lc_batch}, y_batch


def data_provider(args, flag):
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    batch_size = args.batch_size

    if args.task_name == 'classification':
        drop_last = False
        data_set = FluxDataLoader(  # 开创实例, 初始化Data类
            args=args,
            root_path=args.root_path,
            flag=flag,
            encoder=args.encoder,
            # on_multimodal=args.on_multimodal,
            on_mm_statistics = args.on_mm_statistics,
            on_mm_history = args.on_mm_history,
            on_enhance=args.on_enhance
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            # collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)，
            collate_fn=lambda x: collate_fn(x)
        )
        return data_set, data_loader
