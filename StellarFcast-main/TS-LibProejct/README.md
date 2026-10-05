# Virtual Environment Configuration

Simply configure the environment according to `requirements.txt` (we recommend using conda commands to create a new virtual environment, which makes it easier to distinguish from the virtual environment used in LLMProject).

```
pip install -r requirements.txt
```

I must remind you that if you encounter unexpected errors while installing dependencies, I recommend asking a large language model for assistance. All required packages for the experiment have been listed, so unexpected errors should not occur.


# Reproduction

Ensure that the following folders exist in the current directory:

```
├── myDataK20 # （kepler）
├── myDataT20 # （tess）
```

The corresponding reproduction commands (note: the number appended after the script specifies the `CUDA GPU` device index):
```
# base
sh ./scripts/classification/DLinear.sh 1 
sh ./scripts/classification/PatchTST.sh 4
sh ./scripts/classification/iTransformer.sh 2
sh ./scripts/classification/TimesNet.sh 5
sh ./scripts/classification/Informer.sh 3
sh ./scripts/classification/Autoformer.sh 4
sh ./scripts/classification/MICN.sh 4

# with all updates
sh ./scripts/classification/DLinear.sh 1 --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history
sh ./scripts/classification/PatchTST.sh 4 --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history
sh ./scripts/classification/iTransformer.sh 1 --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history
sh ./scripts/classification/TimesNet.sh 1 --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history
sh ./scripts/classification/Informer.sh 1 --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history
sh ./scripts/classification/Autoformer.sh 1 --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history
sh ./scripts/classification/MICN.sh 1 --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history
```

Note: The above reproduction is intended for the Kepler data. If you wish to reproduce results using the TESS data instead, simply locate the corresponding scripts and modify them as follows.
```
example：
# Autoformer.sh：
  --root_path ./myDataK20 \
change to：
  --root_path ./myDataT20 \
```

