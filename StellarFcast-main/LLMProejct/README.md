# Virtual Environment Configuration

Simply configure the environment according to `requirements.txt` (we recommend using conda commands to create a new virtual environment to easily distinguish it from the `TS-LibProject` virtual environment).
```
pip install -r requirements.txt
```

I have to remind you that if you encounter unexpected errors while installing dependencies, I recommend asking a large language model for help. All the packages required for the experiment have been listed, so unexpected errors should not occur.

# Downloading Pretrained Large Language Models

In our experiments, we used the following pretrained large language models. Simply download them from their corresponding `Hugging Face` pages (URLs are listed on the right) to your local machine, as our experiments load the large language models locally.
```
bert_base_uncased # https://huggingface.co/google-bert/bert-base-uncased
gpt2 # https://huggingface.co/openai-community/gpt2
roberta(chinese-roberta-wwm-ext ) # https://huggingface.co/hfl/chinese-roberta-wwm-ext
deberta-v3-base # https://huggingface.co/microsoft/deberta-v3-base
```

Also, we give download script: `./download_models.py`

After downloading, ensure that the following directories exist in your current working directory:

```
├── models
│   ├── bert_base_uncased
│   ├── gpt2 
│   ├── deberta-v3-base
│   └── chinese-roberta-wwm-ext
├── myDataK20 #  （kepelr）
├── myDataT20 #  （tess）
```



# Reproduction
The corresponding reproduction commands are as follows:

```
# (It is recommended to prepend the command with CUDA_VISIBLE_DEVICES=TODO to specify the GPU(s) to use; if omitted, all available GPUs will be used.)# Kepler：
# gpt2 
python main.py --model_type gpt2 --use_lora --exp_num 1  --all --dataset kepler

# bert
python main.py --model_type bert --use_lora --exp_num 1  --all --dataset kepler

# roberta
python main.py --model_type roberta-c --use_lora --exp_num 1  --all --dataset kepler

# deberta
python main.py --model_type deberta --use_lora --exp_num 1  --all --dataset kepler

# FLARE
python main_flare.py --dataset kepler


# TESS :
# gpt2 
python main.py --model_type gpt2 --use_lora --exp_num 1  --all --dataset tess

# bert
python main.py --model_type bert --use_lora --exp_num 1  --all --dataset tess

# roberta
python main.py --model_type roberta-c --use_lora --exp_num 1  --all --dataset tess

# deberta
python main.py --model_type deberta --use_lora --exp_num 1  --all --dataset tess

# FLARE
python main_flare.py --dataset tess
```



Ablation study, used to evaluate the impact of each module on the experimental results.

```
# kepler
# off --on_mm_history
CUDA_VISIBLE_DEVICES=4 python main.py --model_type roberta-c --use_lora --exp_num 1 --on_phy_loss --on_enhance --on_mm_statistics  --dataset kepler

# off --on_mm_statistics
CUDA_VISIBLE_DEVICES=4 python main.py --model_type roberta-c --use_lora --exp_num 1 --on_phy_loss --on_enhance  --on_mm_history --dataset kepler

# off --on_enhance
CUDA_VISIBLE_DEVICES=4 python main.py --model_type roberta-c --use_lora --exp_num 1 --on_phy_loss --on_mm_statistics --on_mm_history --dataset kepler

# off -on_phy_loss
CUDA_VISIBLE_DEVICES=4 python main.py --model_type roberta-c --use_lora --exp_num 1 --on_enhance --on_mm_statistics --on_mm_history --dataset kepler
```



To reproduce `gpt4ts`, you need to run the following commands:

```
# with all:
python main.py --model_type gpt4ts --exp_num 1 --all

# base:
python main.py --model_type gpt4ts --exp_num 1
```



In this way, you can successfully reproduce almost all the experiments mentioned in the paper.

