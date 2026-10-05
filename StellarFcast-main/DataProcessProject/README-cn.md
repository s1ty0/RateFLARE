# 1. 配置环境

和 TS-LibProject的环境一致


# 2. 获取数据

首先，我们需要下载原始数据到`当下目录data文件夹中`，这包括两方面的数据：Kepler和TESS，数据我已经打包好，放在

1. Google: 「https://drive.google.com/drive/folders/1IRCrJeVgdFk8QV48NjXEIcRnBVeqtUgq?usp=sharing」
2. 百度网盘: 「 https://pan.baidu.com/s/1fVw5jf_rhTYBNDs0FRT-DA 提取码: fdc9 」里面

其中Kepler数据引用自 https://huggingface.co/datasets/Maxwell-Jia/kepler_flare， TESS数据来源于我们的数据团队`（现也已经公开）`

注：如果直接在（https://huggingface.co/datasets/Maxwell-Jia/kepler_flare）下载Kepler数据，你将得到原始数据，但我们对原始数据进行了简单的合并处理，用于执行我们后续的数据处理的流程，当然，google和百度网盘里面存放的已经是合并好的数据，直接使用即可。

（合并代码可参考如下）

```
import pandas as pd

# 读取当前目录下所有 parquet 文件并合并保存为 all.parquet
pd.read_parquet("./").to_parquet("all.parquet")
```



获得好数据后，请确保你的目录包含`data文件夹（及文件）`，然后，才可以执行脚本。

```
├── DataProcessProject
│   ├── data
│   │   ├── all.parquet # 对应kepler
│   │   ├── my_data.pt # 对应TESS
...
```



# 3. 执行数据处理脚本

然后，执行对应的数据脚本，进行patch 和clean的操作（同时避免了数据泄露）
在当面目录下依次执行：

```
sh ./data_pipeline_kepler.sh
sh ./data_pipeline_tess.sh
```

将会得到两个文件夹：

```
./myDataK # 存放得到的Kepler数据
./myDataT # 存放得到的TESS数据
```
为了减轻实验的负担量，我们进行了一个数据下采样处理，执行：
```
python createK20.py
python createT20.py
```

将会得到两个文件夹：

```
./myDataK20 # 存放得到的下采样的Kepler数据
./myDataT20 # 存放得到的下采样的TESS数据
```
这样，我们就完成了基础的数据准备工作。

等等！先别急，我们进行实验所必需的，一些改进点的引入，仍需要构建history和statistics的嵌入

首先下载实验所使用的文本编码器：`bert`,

下载脚本：

```
python encoderDown.py
```

下载完后，将会得到如下文件夹：
```
├── textEncoder
│   ├── bert-base-chinese
```

然后，执行如下脚本，得到history和statistics的嵌入：

```
sh ./model_build_emb_pipeline_k.sh
sh ./model_build_emb_pipeline_t.sh
```

随后将这俩文件夹：`myDataK20` 和 `myDataT20`

分别复制、粘贴到`LLMProject文件夹`和`TS-LibProject文件夹`下。这样，我很高兴的向你宣告数据处理章节的结束。

接下来，请转向`LLMProject`和`TS-LibProject`去复现对应的代码。
