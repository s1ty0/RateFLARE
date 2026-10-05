import os, sys, time, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Phase2", "gf"))
import torch, numpy as np
from gf_data import UnifiedDataset, collate
from gf_model import GroundedFlare
from gf_modules import compute_phi, p1_rise_loss, p2_consistency_loss

torch.backends.cudnn.benchmark = True
device = torch.device("cuda")
ds = UnifiedDataset(os.path.join(ROOT, "unified", "kepler", "train"))
sub = np.load(os.path.join(ROOT, "unified", "kepler", "train", "sub_idx10.npy"))
ds = torch.utils.data.Subset(ds, sub)
dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, collate_fn=collate, num_workers=0)

cfg = argparse.Namespace(patch=16, stride=8, d_model=768, pool="cls", encoder_type="llm",
                         use_maskaware=True, use_history=True, use_film=True, use_p1=True, use_p2=True,
                         peft="lora", fake_phys=False, theta=0.0487, lora_r=8, lora_alpha=16,
                         plm_path=os.path.join(ROOT, "StellarFcast-main", "LLMProejct", "models", "roberta-base"))
model = GroundedFlare(cfg).to(device)
opt = torch.optim.AdamW(model.trainable_params(), lr=1e-4)
ce = torch.nn.CrossEntropyLoss()

b = next(iter(dl))
x4, hist = b["x4"].to(device), b["hist"].to(device)
y48, y480, z = b["y48"].to(device), b["y480"].to(device), b["z"].to(device)

def step(use_amp=False):
    if use_amp:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits48, logits480, z_hat = model(x4, hist)
            loss = ce(logits48, y48) + ce(logits480, y480)
            phi = compute_phi(x4[:, 1:2])
            loss = loss + 0.1*p1_rise_loss(z_hat, z, y480) + 0.1*p2_consistency_loss(torch.softmax(logits480,-1)[:,1], phi, torch.tensor(0.0487, device=device))
    else:
        logits48, logits480, z_hat = model(x4, hist)
        loss = ce(logits48, y48) + ce(logits480, y480)
        phi = compute_phi(x4[:, 1:2])
        loss = loss + 0.1*p1_rise_loss(z_hat, z, y480) + 0.1*p2_consistency_loss(torch.softmax(logits480,-1)[:,1], phi, torch.tensor(0.0487, device=device))
    opt.zero_grad(); loss.backward(); opt.step()

# warmup
for _ in range(3): step()

# data load timing (10 batches)
torch.cuda.synchronize(); t0=time.time()
it = iter(dl)
for _ in range(10): next(it)
torch.cuda.synchronize(); print(f"data load: {(time.time()-t0)/10*1000:.1f} ms/batch")

# forward timing
torch.cuda.synchronize(); t0=time.time()
for _ in range(20):
    model(x4, hist)
torch.cuda.synchronize(); print(f"forward FP32: {(time.time()-t0)/20*1000:.1f} ms")

# full step FP32
torch.cuda.synchronize(); t0=time.time()
for _ in range(20): step(False)
torch.cuda.synchronize(); print(f"full step FP32: {(time.time()-t0)/20*1000:.1f} ms")

# full step AMP
torch.cuda.synchronize(); t0=time.time()
for _ in range(20): step(True)
torch.cuda.synchronize(); print(f"full step AMP(BF16): {(time.time()-t0)/20*1000:.1f} ms")
