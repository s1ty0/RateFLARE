# 1. Configure Environment

As same as TS-LibProject's env.

# 2. Obtain Data

First, download the raw data to `the data folder` in the current directory. This includes two datasets: Kepler and TESS. The data has been pre-packaged and is available via:
1. Google: 「https://drive.google.com/drive/folders/1IRCrJeVgdFk8QV48NjXEIcRnBVeqtUgq?usp=sharing」
2. Baidu Netdisk: 「 https://pan.baidu.com/s/1fVw5jf_rhTYBNDs0FRT-DA Extraction Code: fdc9 」

Kepler Data Source: https://huggingface.co/datasets/Maxwell-Jia/kepler_flare， 
TESS Data Source: Our internal data team (now publicly available)

If you download Kepler data directly from HuggingFace, you will get the raw dataset. However, we have performed a simple merging process on the raw data to streamline the subsequent processing pipeline.
The data in Google Drive and Baidu Netdisk is already merged and ready to use.

Code to Merge Raw Data (if needed):

```
import pandas as pd
pd.read_parquet("./").to_parquet("all.parquet")
```

After downloading, ensure your directory includes the `data folder` with the following structure:

```
├── DataProcessProject
│   ├── data
│   │   ├── all.parquet # kepler
│   │   ├── my_data.pt # TESS
...
```



# 3. Execute Data Processing Scripts

Run the following scripts sequentially to perform patching and cleaning operations (avoiding data leakage):

```
sh ./data_pipeline_kepler.sh
sh ./data_pipeline_tess.sh
```

This will generate two folders:

```
./myDataK # Kepler
./myDataT # TESS
```

To reduce data volume, execute the following scripts:

```
python createK20.py
python createT20.py
```

This will generate:

```
./myDataK20 
./myDataT20 
```

To support experimental improvements, we need to build embeddings for history and statistics.
First, download the text encoder`bert`, 

download script:
```
python encoderDown.py
```

After downloading, you will obtain the following directory structure:
```
├── textEncoder
│   ├── bert_base_uncased
```

Then run the following scripts to generate embeddings:

```
sh ./model_build_emb_pipeline_k.sh
sh ./model_build_emb_pipeline_t.sh
```
Copy the `myDataK20` and `myDataT20` folders to the following project directories:
- LLMProject folder
- TS-LibProject folder
Congratulations! The data processing phase is complete.
Next, proceed to the `LLMProject` and `TS-LibProject` folders to reproduce the experiments.