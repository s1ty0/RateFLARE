'# 虚拟环境的配置

按照`requirements.txt`配置即可(推荐用conda的指令新建虚拟环境，方便同`TS-LibProject`的虚拟环境作区分)

```
pip install -r requirements.txt
```

我不得不提醒你的是，如果在安装依赖的时候碰到了一些意外的错误，我推荐你询问大模型寻求帮助，实验所需的所有包已经列出，应该不会出现意外的错误。



# 预训练大语言模型的下载

我们实验中用了以下预训练大语言模型，在对应的`huggingface`下载到本地即可(对应的网址已经罗列在右面)，因为实验中采用的是从本地加载大语言模型的方式。

```
bert_base_uncased # https://huggingface.co/google-bert/bert-base-uncased
gpt2 # https://huggingface.co/openai-community/gpt2
roberta(chinese-roberta-wwm-ext ) # https://huggingface.co/hfl/chinese-roberta-wwm-ext
deberta-v3-base # https://huggingface.co/microsoft/deberta-v3-base
```

同样，我们提供了下载脚本： `./download_models.py`

下载好后，确保当下目录存在以下文件夹：

```
├── models
│   ├── bert_base_uncased
│   ├── gpt2 
│   ├── deberta-v3-base
│   └── chinese-roberta-wwm-ext
├── myDataK20 # 提前构建好的数据集存放目录 （kepelr）
├── myDataT20 # 提前构建好的数据集存放目录 （tess）
```



# 复现

对应的复现指令如下：

```
# 以下是全部改进点开启后的复现指令：（建议在指令前加上：CUDA_VISIBLE_DEVICES=TODO, 来指明所用的GPU，若不加，代表使用所有可用的GPU）
# Kepler数据上：
# gpt2 模型
python main.py --model_type gpt2 --use_lora --exp_num 1  --all --dataset kepler

# bert模型
python main.py --model_type bert --use_lora --exp_num 1  --all --dataset kepler

# roberta模型
python main.py --model_type roberta-c --use_lora --exp_num 1  --all --dataset kepler

# deberta模型
python main.py --model_type deberta --use_lora --exp_num 1  --all --dataset kepler

# 尝试复现的FLARE
python main_flare.py --dataset kepler


# TESS 数据上：
# gpt2 模型
python main.py --model_type gpt2 --use_lora --exp_num 1  --all --dataset tess

# bert模型
python main.py --model_type bert --use_lora --exp_num 1  --all --dataset tess

# roberta模型
python main.py --model_type roberta-c --use_lora --exp_num 1  --all --dataset tess

# deberta模型
python main.py --model_type deberta --use_lora --exp_num 1  --all --dataset tess

# 尝试复现的FLARE
python main_flare.py --dataset tess
```



消融实验，用来判定各个模块对实验的影响

```
# kepler数据
# 关闭--on_mm_history
CUDA_VISIBLE_DEVICES=4 python main.py --model_type roberta-c --use_lora --exp_num 1 --on_phy_loss --on_enhance --on_mm_statistics  --dataset kepler

# 关闭 --on_mm_statistics
CUDA_VISIBLE_DEVICES=4 python main.py --model_type roberta-c --use_lora --exp_num 1 --on_phy_loss --on_enhance  --on_mm_history --dataset kepler

# 关闭--on_enhance
CUDA_VISIBLE_DEVICES=4 python main.py --model_type roberta-c --use_lora --exp_num 1 --on_phy_loss --on_mm_statistics --on_mm_history --dataset kepler

# 关闭--on_phy_loss
CUDA_VISIBLE_DEVICES=4 python main.py --model_type roberta-c --use_lora --exp_num 1 --on_enhance --on_mm_statistics --on_mm_history --dataset kepler
```



复现`gpt4ts`，需要执行如下指令:

```
# 改进点加入后(all):
python main.py --model_type gpt4ts --exp_num 1 --all

# 改进点未加入之前（base）:
python main.py --model_type gpt4ts --exp_num 1
```



如此你可以成功复现论文中提到的几乎全部实验。

