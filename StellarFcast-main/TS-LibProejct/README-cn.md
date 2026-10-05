# 虚拟环境的配置

按照`requirements.txt`配置即可(推荐用conda的指令新建虚拟环境，方便同`LLMProject`的虚拟环境作区分)

```
pip install -r requirements.txt
```

我不得不提醒你的是，如果在安装依赖的时候碰到了一些意外的错误，我推荐你询问大模型寻求帮助，实验所需的所有包已经列出，应该不会出现意外的错误。



# 复现

确保当下目录存在以下文件夹：

```
├── myDataK20 # 提前构建好的数据集存放目录（kepler）
├── myDataT20 # 提前构建好的数据集存放目录（tess）
```


对应的复现指令（注：脚本后面携带的数字，代表指定的`cuda gpu`序号）：

```
# 基础运行
sh ./scripts/classification/DLinear.sh 1 
sh ./scripts/classification/PatchTST.sh 4
sh ./scripts/classification/iTransformer.sh 2
sh ./scripts/classification/TimesNet.sh 5
sh ./scripts/classification/Informer.sh 3
sh ./scripts/classification/Autoformer.sh 4
sh ./scripts/classification/MICN.sh 4

# 改进开启后运行：
sh ./scripts/classification/DLinear.sh 1 --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history --itr 3
sh ./scripts/classification/PatchTST.sh 4 --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history
sh ./scripts/classification/iTransformer.sh 1 --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history
sh ./scripts/classification/TimesNet.sh 1 --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history
sh ./scripts/classification/Informer.sh 1 --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history
sh ./scripts/classification/Autoformer.sh 1 --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history
sh ./scripts/classification/MICN.sh 1 --on_phy_loss --on_enhance --on_mm_statistics --on_mm_history
```

注：以上的复现针对Kepler数据而言，如果想复现对应的TESS数据，只需要找到对应的脚本，按如下修改即可。

```
比如：
# Autoformer.sh中：
  --root_path ./myDataK20 \
改为：
  --root_path ./myDataT20 \
```

