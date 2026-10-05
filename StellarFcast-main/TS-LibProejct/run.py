import argparse
import importlib
import torch.backends
from utils.print_args import print_args
import random
import time
import numpy as np

if __name__ == "__main__":
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="Our stellar model")  #  UPDATE

    # basic config
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        default="long_term_forecast",  #  长期预测和短期预测的区别是？
        help="task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]",
    )
    parser.add_argument(
        "--is_training", type=int, required=True, default=1, help="status"
    )
    parser.add_argument(
        "--model_id", type=str, required=True, default="test", help="model id"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="Autoformer",
        help="model name, options: [Autoformer, Transformer, TimesNet]",
    )

    # tessDataset loader
    parser.add_argument(
        "--data", type=str, required=True, default="ETTh1", help="dataset type"
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="./myDataK",
        help="root path of the tessDataset file",
    )  #
    parser.add_argument(
        "--data_path", type=str, default="our.npy", help="tessDataset file"
    )
    parser.add_argument(
        "--features",
        type=str,
        default="M",  #  体现在图像上是如何解释
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
    )
    parser.add_argument(
        "--target", type=str, default="OT", help="target feature in S or MS task"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",  #  时间特征编码 ？
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument(
        "--pred_len", type=int, default=96, help="prediction sequence length"
    )
    parser.add_argument(
        "--seasonal_patterns", type=str, default="Monthly", help="subset for M4"
    )
    parser.add_argument(
        "--inverse",
        action="store_true",
        help="inverse output tessDataset",
        default=False,
    )

    # inputation task
    parser.add_argument("--mask_rate", type=float, default=0.25, help="mask ratio")

    # anomaly detection task
    parser.add_argument(
        "--anomaly_ratio", type=float, default=0.25, help="prior anomaly ratio (%%)"
    )

    # model define
    parser.add_argument(
        "--expand", type=int, default=2, help="expansion factor for Mamba"
    )
    parser.add_argument(
        "--d_conv", type=int, default=4, help="conv kernel size for Mamba"
    )
    parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")
    parser.add_argument("--num_kernels", type=int, default=6, help="for Inception")
    parser.add_argument("--enc_in", type=int, default=1, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument(
        "--moving_avg", type=int, default=25, help="window size of moving average"
    )
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument(
        "--distil",
        action="store_false",
        help="whether to use distilling in encoder, using this argument means not using distilling",
        default=True,
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--channel_independence",
        type=int,
        default=1,
        help="0: channel dependence 1: channel independence for FreTS model",
    )
    parser.add_argument(
        "--decomp_method",
        type=str,
        default="moving_avg",
        help="method of series decompsition, only support moving_avg or dft_decomp",
    )
    parser.add_argument(
        "--use_norm",
        type=int,
        default=1,
        help="whether to use normalize; True 1 False 0",
    )
    parser.add_argument(
        "--down_sampling_layers",
        type=int,
        default=0,
        help="num of down sampling layers",
    )
    parser.add_argument(
        "--down_sampling_window", type=int, default=1, help="down sampling window size"
    )
    parser.add_argument(
        "--down_sampling_method",
        type=str,
        default=None,
        help="down sampling method, only support avg, max, conv",
    )
    parser.add_argument(
        "--seg_len",
        type=int,
        default=96,
        help="the length of segmen-wise iteration of SegRNN",
    )

    # optimization
    parser.add_argument(
        "--num_workers", type=int, default=4, help="tessDataset loader num workers"
    )  # 本地默认是0， 云服务器要改， 我先修改cheng0,（试试），我怀疑是这里导致dataloader加载不了，从10改成1. 果然
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=100, help="train epochs")  #
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size of train input tessDataset",
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="early stopping patience"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="optimizer learning rate"
    )  # 已按照FLARE文章修改
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument(
        "--lradj", type=str, default="type1", help="adjust learning rate"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )

    # GPU
    parser.add_argument(
        "--use_gpu", type=bool, default=True, help="use gpu"
    )  #  服务器训练需更改成 True
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--gpu_type", type=str, default="cuda", help="gpu type"
    )  # cuda or mps
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0,1", help="device ids of multile gpus"
    )

    # de-stationary projector params
    parser.add_argument(
        "--p_hidden_dims",
        type=int,
        nargs="+",
        default=[128, 128],
        help="hidden layer dimensions of projector (List)",
    )
    parser.add_argument(
        "--p_hidden_layers",
        type=int,
        default=2,
        help="number of hidden layers in projector",
    )

    # metrics (dtw)
    parser.add_argument(
        "--use_dtw",
        type=bool,
        default=False,
        help="the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)",
    )

    # Augmentation
    parser.add_argument(
        "--augmentation_ratio", type=int, default=0, help="How many times to augment"
    )
    parser.add_argument("--seed", type=int, default=2, help="Randomization seed")
    parser.add_argument(
        "--jitter",
        default=False,
        action="store_true",
        help="Jitter preset augmentation",
    )
    parser.add_argument(
        "--scaling",
        default=False,
        action="store_true",
        help="Scaling preset augmentation",
    )
    parser.add_argument(
        "--permutation",
        default=False,
        action="store_true",
        help="Equal Length Permutation preset augmentation",
    )
    parser.add_argument(
        "--randompermutation",
        default=False,
        action="store_true",
        help="Random Length Permutation preset augmentation",
    )
    parser.add_argument(
        "--magwarp",
        default=False,
        action="store_true",
        help="Magnitude warp preset augmentation",
    )
    parser.add_argument(
        "--timewarp",
        default=False,
        action="store_true",
        help="Time warp preset augmentation",
    )
    parser.add_argument(
        "--windowslice",
        default=False,
        action="store_true",
        help="Window slice preset augmentation",
    )
    parser.add_argument(
        "--windowwarp",
        default=False,
        action="store_true",
        help="Window warp preset augmentation",
    )
    parser.add_argument(
        "--rotation",
        default=False,
        action="store_true",
        help="Rotation preset augmentation",
    )
    parser.add_argument(
        "--spawner",
        default=False,
        action="store_true",
        help="SPAWNER preset augmentation",
    )
    parser.add_argument(
        "--dtwwarp",
        default=False,
        action="store_true",
        help="DTW warp preset augmentation",
    )
    parser.add_argument(
        "--shapedtwwarp",
        default=False,
        action="store_true",
        help="Shape DTW warp preset augmentation",
    )
    parser.add_argument(
        "--wdba",
        default=False,
        action="store_true",
        help="Weighted DBA preset augmentation",
    )
    parser.add_argument(
        "--discdtw",
        default=False,
        action="store_true",
        help="Discrimitive DTW warp preset augmentation",
    )
    parser.add_argument(
        "--discsdtw",
        default=False,
        action="store_true",
        help="Discrimitive shapeDTW warp preset augmentation",
    )
    parser.add_argument("--extra_tag", type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument("--patch_len", type=int, default=16, help="patch length")

    # 配置项从以下开始看，以上用处很小
    # 是否开启多模态模式：
    # 是否开启多模态模式：use_multimodal, 其对应两类：（统计信息+历史序列），分别对应--on_mm_statistics、--on_mm_history
    parser.add_argument(
        "--use_multimodal",
        action="store_true",
        help="Enable multimodal input (x_enc + text_emb[stastic])",
    )
    parser.add_argument(
        "--on_mm_statistics",
        action="store_true",
        help="Enable multimodal input (x_enc + text_emb[stastic])",
    )
    parser.add_argument(
        "--on_mm_history",
        action="store_true",
        help="Enable multimodal input (x_enc + text_emb[history])",
    )

    # 若开启多模态，则选择文本嵌入模型
    parser.add_argument(
        "--encoder", type=str, default="bert-chinese", help="type of encoder we use."
    )
    parser.add_argument(
        "--text_emb_dim", type=int, default=384, help="type of encoder we use."
    )  #  指定其特征维度
    # 是否开启单模态特征增强
    parser.add_argument(
        "--on_enhance", action="store_true", help="Enable flux augmentation(Add 差分)"
    )
    # 是否开启物理损失函数约束
    parser.add_argument(
        "--on_phy_loss", action="store_true", help="Enable physical loss"
    )

    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device("cuda:{}".format(args.gpu))
        print("Using GPU")
    else:
        if hasattr(torch.backends, "mps"):
            args.device = (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        else:
            args.device = torch.device("cpu")
        print("Using cpu or mps")

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.on_mm_statistics or args.on_mm_history:
        args.use_multimodal = True

    # 根据创新点选择输入维度
    if args.on_enhance and args.on_mm_statistics and args.on_mm_history:
        args.enc_in = 4
    elif args.on_enhance and args.on_mm_statistics:
        args.enc_in = 3
    elif args.on_enhance and args.on_mm_history:
        args.enc_in = 3
    elif args.on_mm_statistics and args.on_mm_history:
        args.enc_in = 3
    elif args.on_mm_statistics or args.on_mm_history or args.on_enhance:
        args.enc_in = 2

    # 根据所选的模型，自动设置其输入维度
    ENCODER_DIM_MAP = {
        "minLM": 384,
        "bert-chinese": 768,
        # 未来可加： "bge": "./textEncoder/bge-small-en-v1.5", ...
    }
    args.text_emb_dim = ENCODER_DIM_MAP[args.encoder]

    module_name = "exp.exp_classification"
    exp_module = importlib.import_module(module_name)
    Exp_Classification = exp_module.Exp_Classification

    print("Args in experiment:")
    print_args(args)

    # >>> 新增：关键配置高亮 <<<
    print("\n" + "=" * 60)
    print("🔑 Key Experimental Settings:")
    print(
        f"  ➤ Multimodal (text[statistics] + LC):              {'✅ ON' if args.on_mm_statistics else '❌ OFF'}"
    )
    print(
        f"  ➤ Multimodal-history (text[history] + LC):              {'✅ ON' if args.on_mm_history else '❌ OFF'}"
    )
    print(
        f"  ➤ Time Series Enhancement (Δflux):     {'✅ ON' if args.on_enhance else '❌ OFF'}"
    )
    print(
        f"  ➤ Physics-Regularized Loss:            {'✅ ON' if args.on_phy_loss else '❌ OFF'}"
    )
    print("=" * 60 + "\n")

    Exp = Exp_Classification

    if args.is_training:
        for ii in range(args.itr):
            _seed = args.seed + ii
            random.seed(_seed)
            np.random.seed(_seed)
            torch.manual_seed(_seed)
            # setting record of experiments
            exp = Exp(args)  # set experiments

            # 构建关键特性标签
            features_tags = []
            if args.on_mm_statistics:
                features_tags.append("1MMs")  # MultiModal
            if args.on_phy_loss:
                features_tags.append("2PHY")  # Physics loss
            if args.on_mm_history:
                features_tags.append("3MMh")  # MultiModal
            if args.on_enhance:
                features_tags.append("4ENH")  # Enhancement

            feature_str = "_".join(features_tags) if features_tags else "BASE"

            time_str = time.strftime("%Y%m%d_%H%M%S")

            # 精简版 setting（保留最关键的信息，并加入时间戳）
            setting = (
                f"{args.task_name}_{args.model}_"
                f"sl{args.seq_len}_"
                f"{feature_str}_"
                f"{args.des}_"
                f"{time_str}_"
                f"{ii}"
            )

            print(
                ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting)
            )
            exp.train(setting)

            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
            )
            exp.test(setting)
            if args.gpu_type == "mps":
                torch.backends.mps.empty_cache()
            elif args.gpu_type == "cuda":
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0

        features_tags = []
        if args.on_mm_statistics:
            features_tags.append("1MMs")  # MultiModal
        if args.on_phy_loss:
            features_tags.append("2PHY")  # Physics loss
        if args.on_mm_history:
            features_tags.append("3MMh")  # MultiModal
        if args.on_enhance:
            features_tags.append("4ENH")  # Enhancement

        feature_str = "_".join(features_tags) if features_tags else "BASE"

        # 精简版 setting（保留最关键的信息）
        setting = (
            f"{args.task_name}_{args.model}_"
            f"sl{args.seq_len}_"
            f"{feature_str}_"
            f"{args.des}_{ii}"
        )

        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting, test=1)
        if args.gpu_type == "mps":
            torch.backends.mps.empty_cache()
        elif args.gpu_type == "cuda":
            torch.cuda.empty_cache()
