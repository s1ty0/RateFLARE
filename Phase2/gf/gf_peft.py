# -*- coding: utf-8 -*-
"""Phase 2.6 PEFT 变体: none / bitfit / partial / lora / adapter / prefix / ptuning

- lora: 手动实现 (gf_modules.apply_lora, 零依赖);
- none / bitfit / partial: 手写, 零依赖;
- adapter / prefix / ptuning: 走 peft 库 (需 peft>=0.7, task_type=FEATURE_EXTRACTION);
  peft 缺失或版本不兼容时报错 — 立即回传报错信息, 我做版本适配 (或临时跳过该变体)。
"""


def _layer_module_list(backbone):
    for cand in ("encoder.layer", "encoder.block", "h", "transformer.layer", "layer"):
        obj = backbone
        for part in cand.split("."):
            try:
                obj = getattr(obj, part)
            except AttributeError:
                obj = None
                break
        if obj is not None and hasattr(obj, "__len__"):
            return obj
    return None


def _freeze_all(backbone):
    for p in backbone.parameters():
        p.requires_grad_(False)


def apply_peft(model, cfg):
    """把 cfg.peft 对应的微调策略施加到 model.backbone (嵌入/历史/头永远可训练)。"""
    backbone = model.backbone
    method = cfg.peft
    _freeze_all(backbone)

    if method == "none":
        print("[PEFT] none: 骨干全冻结 (仅训输入投影/历史/头)")
    elif method == "bitfit":
        n = 0
        for name, p in backbone.named_parameters():
            if "bias" in name:
                p.requires_grad_(True)
                n += 1
        print(f"[PEFT] bitfit: 可训练偏置 {n} 个")
    elif method == "partial":
        layers = _layer_module_list(backbone)
        assert layers is not None, "未找到层模块列表, partial 不支持该骨干"
        n = 0
        for blk in layers[-cfg.partial_layers:]:
            for p in blk.parameters():
                p.requires_grad_(True)
                n += 1
        print(f"[PEFT] partial: 解冻末 {cfg.partial_layers} 层 ({n} 个参数)")
    elif method == "lora":
        from gf_modules import apply_lora
        n = apply_lora(backbone, r=cfg.lora_r, alpha=cfg.lora_alpha)
        print(f"[PEFT] lora: 包装 {n} 个 Q/K/V 投影 (0 表示该骨干无分离式 QKV, 等同 none)")
    elif method in ("adapter", "prefix", "ptuning"):
        try:
            from peft import (get_peft_model, PrefixTuningConfig,
                              PromptEncoderConfig, TaskType)
        except ImportError:
            raise RuntimeError(f"{method} 需要 peft 库 (pip install peft); 或改用 none/bitfit/partial/lora")
        if method == "adapter":
            # peft>=0.9 已移除 AdapterConfig; 改用 IA3 近似瓶颈 adapter 或降级 peft<0.9
            try:
                from peft import IA3Config as _AdapterCfg
                conf = _AdapterCfg(task_type=TaskType.FEATURE_EXTRACTION)
                print("[PEFT] adapter: peft>=0.9 无 AdapterConfig, 用 IA3 近似 (或 pip install 'peft<0.9')")
            except ImportError:
                raise RuntimeError("adapter 需要 peft<0.9 (AdapterConfig 已移除); 或改用 --peft ia3")
        elif method == "prefix":
            conf = PrefixTuningConfig(num_virtual_tokens=cfg.prefix_len,
                                      task_type=TaskType.FEATURE_EXTRACTION)
        else:
            conf = PromptEncoderConfig(num_virtual_tokens=cfg.ptuning_tokens,
                                       task_type=TaskType.FEATURE_EXTRACTION)
        model.backbone = get_peft_model(backbone, conf)
        print(f"[PEFT] {method}: peft 包装完成 (如报错请回传完整 traceback)")
    else:
        raise ValueError(f"未知 peft 方法: {method}")

    n = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"[PEFT] 骨干可训练参数: {n / 1e6:.3f}M")
    return method
