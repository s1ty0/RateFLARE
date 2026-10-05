# RateFLARE

Code for stellar flare forecasting with historical activity rates, light curves, and a unified evaluation pipeline for Kepler and TESS. The neural implementation retains its development name, `GroundedFlare`, in `Phase2/gf/`.

## Repository layout

| Directory | Contents |
| --- | --- |
| `Phase0/` | Data and baseline audits |
| `Phase1/` | Unified preprocessing, event export, baseline adapters, and metrics |
| `Phase2/gf/` | Neural model, activity-rate features, optional point-process branch, and training |
| `Phase3/`, `Phase4/` | Robustness, stratification, and attribution analysis |
| `Phase5/` | Table export |
| `tools/` | Rate baselines, ablations, significance analysis, and experiment launchers |
| `figures/` | Paper figure generation |
| `FLARE-main/`, `StellarFcast-main/` | Baseline source snapshots used in this project |

## Installation

Use Python 3.10 on Linux; a CUDA-capable GPU is recommended for neural training. Shell launchers require Bash. From the repository root:

```bash
git clone https://github.com/s1ty0/RateFLARE.git
cd RateFLARE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers the main pipeline. Baseline projects retain their own requirements and setup instructions; install them in separate environments when needed. Install a PyTorch build appropriate for your CUDA runtime.

## Data

Download and extract the original Kepler and TESS data into `data/` in the repository root. Raw `.npy` and `.pt` files, including `all.pt`, are distributed separately through the existing links:

1. [Google Drive](https://drive.google.com/drive/folders/1IRCrJeVgdFk8QV48NjXEIcRnBVeqtUgq?usp=sharing)
2. [Baidu Netdisk](https://pan.baidu.com/s/1fVw5jf_rhTYBNDs0FRT-DA) — extraction code: **fdc9**

Kepler source: [Maxwell-Jia/kepler_flare on Hugging Face](https://huggingface.co/datasets/Maxwell-Jia/kepler_flare). TESS was prepared by the original data team and is distributed through the links above.

Use the file appropriate for each loader: the unified Kepler builder reads merged `all.parquet`; the TESS builder reads a `.pt` dictionary containing `flux` and `mask` (called `my_data.pt` in the baseline instructions). Pass the actual filename from your download; a `.npy` or `.pt` file is not interchangeable with Kepler Parquet. The original preprocessing instructions are in [DataProcessProject/README.md](StellarFcast-main/DataProcessProject/README.md).

Data, generated splits, downloaded model weights, predictions, and checkpoints are excluded from Git.

## Quick start: Kepler

Run the following from the repository root after placing `all.parquet` in `data/`:

```bash
# Build 512-point inputs and 48-/480-point forecast labels.
python Phase1/build_unified.py --dataset kepler \
  --kepler_parquet data/all.parquet --out unified

# Export causal event histories and activity-rate inputs.
python Phase1/export_events.py --kepler_parquet data/all.parquet \
  --unified unified --cad_min 29.4244

# Generate the fixed 25% training subset used by experiment launchers.
python tools/gen_subidx.py --unified unified --dataset kepler --frac 0.25

# Download backbones separately (including RoBERTa).
python tools/dl_models.py

# Main rate-aware configuration: no stellar metadata, no point-process branch.
python Phase2/gf/train_gf.py --dataset kepler --unified unified \
  --plm_path StellarFcast-main/LLMProejct/models/roberta-base \
  --out Phase2 --train_sub_idx unified/kepler/train/sub_idx25.npy \
  --no_meta --seed 42 --epochs 150 --patience 20 --batch 64 \
  --theta 0.0487 --cad_min 29.4244 --select_metric auprc480
```

Activity-rate features are enabled by default; `--no_rate` is the paired ablation and `--use_pp` enables the optional point-process branch. `--theta 0.0487` is the existing Kepler experiment setting; omit it to estimate the consistency threshold from training data. Repeat with seeds `2025` and `777` for the three-seed configuration.

For TESS preprocessing, use:

```bash
python Phase1/build_unified.py --dataset tess \
  --tess_pt data/my_data.pt --out unified
```

Replace `data/my_data.pt` with the downloaded TESS filename as needed. `export_events.py` currently targets Kepler; the rate-aware Kepler workflow requires its `events.npz` outputs. TESS data without stellar IDs are split by light-curve record.

## Evaluation

The main configuration above writes predictions under `Phase2/preds/kepler/`:

```bash
python Phase1/eval_predictions.py --dataset kepler --unified unified \
  --preds_dir Phase2/preds \
  --model_id gf_roberta-base_lora_noM0_M1_M2_M6_noPP_P1P2_RATE_s42

python tools/make_rate_baseline_preds.py --dataset kepler \
  --unified unified --preds Phase2/preds --cad_min 29.4244

python Phase1/eval_predictions.py --dataset kepler --unified unified \
  --preds_dir Phase2/preds --model_id bl_rate_all
```

The pipeline reports both forecast horizons on natural and balanced test sets, including AUPRC, AUROC, F1, Brier score, and ECE. Decision thresholds are selected on validation data. Kepler splitting groups records by stellar ID with seed 42; the training subset also uses seed 42.

`tools/w5_final.sh` contains the main three-seed and ablation configurations; historical launchers may require the documented Conda environment and multiple GPUs. The figure script reads regenerated predictions and includes some fixed values from the original experiment records. This release packages source code; training and experiments were not rerun during publication.

## Acknowledgements

This project builds on FLARE and StellarF. The bundled baseline snapshots include local adaptations; their original README files and existing license notices are retained. FLARE's MIT license is in [FLARE-main/LICENSE](FLARE-main/LICENSE). Please cite the original work when using the corresponding baseline components.
