import os, sys
import torch, transformers, sklearn, numpy
print("deps ok:", "torch", torch.__version__, "| transformers", transformers.__version__, "| numpy", numpy.__version__)
try:
    import peft
    print("peft", peft.__version__)
except Exception as e:
    print("peft FAIL:", e)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Phase2", "gf"))
import gf_modules, gf_model, gf_peft, gf_data
print("gf imports ok")

from gf_data import UnifiedDataset, collate
ds = UnifiedDataset(os.path.join(ROOT, "unified", "kepler", "train"))
print("dataset len", len(ds))
b = ds[0]
print("x4", tuple(b["x4"].shape), "hist", tuple(b["hist"].shape),
      "meta", tuple(b["meta"].shape), "y48", b["y48"].item(), "y480", b["y480"].item(),
      "z", round(float(b["z"]), 5))
print("ALL_CHECKS_PASS")
