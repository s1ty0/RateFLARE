import argparse
import os
from argparse import Namespace

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ARGS:
    task_name = "classification"
    patch_size = 16
    stride = 8
    gpt_layers = 3
    d_ff = 256
    dropout = 0.1
    seed = 321
    droprate_data = 0.2
    model_dir_name = "gpt4tsV2"
    model_name = "gpt_allzi"
    seq_len = 512
    horizon_len = 48
    p2n_rate_train = -1
    p2n_rate_test = 1
    data_stride_len_train = 100
    data_stride_len_test = 100
    batch_size = 128
    llm_model = "GPT2"
    tslm_model = "GPT2"
    n_runs = 5
    timestamp_type = "fixed"
    if_test_save = False
    out_proj = "First"
    if_addtoken_grad = False
    if_only_metavec = False
    ifAdapter = False

    # important
    ifP_tuning = False
    ifLoRA = False

    adam_epsilon = 1e-08
    learning_rate = 1e-05
    # lr_scheduler
    ifschedule_with_warmup = True
    schedule_type = "LambdaLR"
    # vec->str
    ifReprogram = False
    Reprogram_type = "ATT"
    # if_decompose
    if_decompose = True
    decompose_type = "Simple"
    # ifAdapter
    ifAdapter = False

    if_dont_use_id = False
    ifNoNorm = False
    if_use_his01 = False
    if_use_meta = False
    patience = 15
    MAX_epoches = 200
    npy_dir_path = (
        os.path.join(_BASE_DIR, "data", "seqlen_512_horizon_48_stride_50_noPeriod")
    )
    dir_path = _BASE_DIR
    kids_path = os.path.join(_BASE_DIR, "data", "kids.npy")
    meta_feats_path = os.path.join(_BASE_DIR, "data", "kepler_meta_feats3.npy")
    num_classes = 2
    dim_in = 2
    dim_meta_1 = 35
    dim_meta_0 = 3000
    dim_hid = 128
    d_model = 768

    @classmethod
    def _get_class_variables(cls):
        return {
            k: v
            for k, v in vars(cls).items()
            if not callable(v) and not k.startswith("_")
        }

    @classmethod
    def _update_class_variables(cls, updates):
        if updates is None:
            return
        if isinstance(updates, Namespace):
            updates = updates.__dict__
        for k, v in updates.items():
            setattr(cls, k, v)
