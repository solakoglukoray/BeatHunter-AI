"""
PANNs fine-tuning — unfreezes the AudioSet encoder and trains it on our data.

Where this starts from. The frozen encoder + PCA-128 + LogReg reaches, on 2,847 held-out
tracks (GroupShuffleSplit by track, random_state=42):

    artist top-1 12.96%  top-5 32.03%  (22.9x chance)   genre 24.17%

That is the bar. Fine-tuning has to beat it to be worth anything, and the same held-out
tracks are used here so the numbers stay comparable.

    python panns_finetune.py --smoke          # tiny run, proves the machinery
    python panns_finetune.py --kill-test      # same, for the SIGKILL drill
    python panns_finetune.py                  # full run
    python panns_finetune.py --resume         # continue after an interruption

CRASH TOLERANCE — read before changing anything here
────────────────────────────────────────────────────
This laptop is carried daily and has died mid-job. A multi-hour run WILL be interrupted,
so the checkpoint has to carry everything needed to continue, not just weights:

    model, optimizer, scheduler, epoch, global step, best score, and the RNG states

Missing any one of those makes a resume silently different from continuing: without
optimizer state Adam restarts cold, without the scheduler the LR jumps back to its
initial value, without RNG state the augmentation stream repeats.

Write order is model-then-marker and both go through os.replace, so an interrupted write
leaves the previous checkpoint intact rather than a truncated file. A failed checkpoint
warns and lets training continue — it must never be the thing that ends a long run — but
the warning says plainly that protection is off.

Verified by SIGKILL, not by the presence of a file. See --kill-test.
"""

import os, sys, time, json, math, argparse, random, warnings
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, librosa, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

import thermal_guard

# ---- paths ----
CKPT_PATH   = "models/panns_finetune_checkpoint.pt"
BEST_PATH   = "models/panns_finetune_best.pt"
STATE_PATH  = "models/panns_finetune_state.json"
TRACK_META  = "models/vector_meta_track.csv"
SLICE_META  = "dataset_processed/metadata_processed.csv"
TAXONOMY    = "models/taxonomy.json"
TEMP_LOG    = os.path.expanduser("~/bench_out/temps_finetune.csv")

# ---- audio / training ----
SR          = 32000       # PANNs' native rate; anything else degrades the encoder
TRAIN_SECS  = 10          # random crop per sample — cheaper than 30s and augments
EVAL_SECS   = 30
BATCH       = 8
EPOCHS      = 12
LR_ENCODER  = 1e-4        # the pretrained encoder wants a gentler rate than fresh heads
LR_HEAD     = 1e-3
CKPT_EVERY_S = 90
WORKERS     = 4


# ============================================================
# MODEL
# ============================================================
class PannsClassifier(nn.Module):
    """Cnn14 trunk with fresh artist/genre heads on its 2048-dim embedding."""

    def __init__(self, n_artists, n_genres, ckpt_path):
        super().__init__()
        from panns_inference.models import Cnn14
        self.trunk = Cnn14(sample_rate=SR, window_size=1024, hop_size=320, mel_bins=64,
                           fmin=50, fmax=14000, classes_num=527)
        if ckpt_path and os.path.exists(ckpt_path):
            self.trunk.load_state_dict(
                torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"])
        self.drop = nn.Dropout(0.3)
        self.artist = nn.Linear(2048, n_artists)
        self.genre = nn.Linear(2048, n_genres)

    def forward(self, x):
        e = self.drop(self.trunk(x)["embedding"])
        return self.artist(e), self.genre(e)


# ============================================================
# DATA
# ============================================================
class TrackAudio(Dataset):
    def __init__(self, df, seconds, train):
        self.df, self.n, self.train = df.reset_index(drop=True), int(SR * seconds), train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:
            y, _ = librosa.load(r["path"], sr=SR, duration=EVAL_SECS)
        except Exception:
            y = np.zeros(self.n, dtype=np.float32)
        if len(y) < self.n:
            y = np.pad(y, (0, self.n - len(y)))
        elif self.train:
            s = random.randint(0, len(y) - self.n)      # random crop = augmentation
            y = y[s:s + self.n]
        else:
            y = y[: self.n]
        return torch.from_numpy(np.asarray(y, dtype=np.float32)), int(r["ya"]), int(r["yg"])


def build_frames(limit_artists=0, limit_per_artist=0):
    meta = pd.read_csv(TRACK_META)
    meta["path"] = meta["track"].map(lambda t: f"dataset/audio/{t}.mp3")
    meta = meta[meta["path"].map(os.path.exists)].reset_index(drop=True)

    if limit_artists or limit_per_artist:
        top = meta["artist_type"].value_counts().head(limit_artists or 999).index
        meta = (meta[meta["artist_type"].isin(top)]
                .groupby("artist_type", group_keys=False)
                .head(limit_per_artist or 10**9).reset_index(drop=True))

    sl = pd.read_csv(SLICE_META)
    sl["track"] = sl["filename"].str.split("_slice").str[0]
    _, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                   random_state=42).split(sl, groups=sl["track"]))
    held = set(sl.iloc[va]["track"].unique())

    artists = np.sort(meta["artist_type"].unique())
    genres = np.sort(meta["genre"].unique())
    a_idx = {a: i for i, a in enumerate(artists)}
    g_idx = {g: i for i, g in enumerate(genres)}
    meta["ya"] = meta["artist_type"].map(a_idx)
    meta["yg"] = meta["genre"].map(g_idx)

    is_val = meta["track"].isin(held)
    return meta[~is_val], meta[is_val], artists, genres, a_idx, g_idx


# ============================================================
# CHECKPOINTING
# ============================================================
def save_checkpoint(path, model, opt, sched, epoch, step, best, scaler=None):
    """
    Atomic full-state save. Includes RNG so the augmentation stream continues rather
    than restarting — otherwise a resumed run sees different crops than a continuous one.
    """
    payload = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict() if sched else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch, "step": step, "best": best,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)          # atomic on the same filesystem


def load_checkpoint(path, model, opt, sched, scaler=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    if sched and ck.get("scheduler"):
        sched.load_state_dict(ck["scheduler"])
    if scaler and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    r = ck.get("rng") or {}
    try:
        if r.get("python"):
            random.setstate(r["python"])
        if r.get("numpy") is not None:
            np.random.set_state(r["numpy"])
        if r.get("torch") is not None:
            torch.set_rng_state(r["torch"].cpu() if hasattr(r["torch"], "cpu") else r["torch"])
        if r.get("cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() if hasattr(s, "cpu") else s for s in r["cuda"]])
    except Exception as e:
        print(f"  ! RNG durumu geri yuklenemedi ({e}) — augmentation akisi farkli olacak")
    return ck["epoch"], ck["step"], ck["best"]


def write_state(epoch, step, best, note=""):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"epoch": epoch, "step": step, "best": best,
                   "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note}, f, indent=2)
    os.replace(tmp, STATE_PATH)


# ============================================================
# EVALUATION
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, g_of_a, use_bf16=False):
    model.eval()
    a_ok = a5_ok = g_ok = n = 0
    for x, ya, yg in loader:
        x, ya, yg = x.to(device), ya.to(device), yg.to(device)
        if use_bf16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                la, lg = model(x)
        else:
            la, lg = model(x)
        la, lg = la.float(), lg.float()
        a_ok += (la.argmax(1) == ya).sum().item()
        a5_ok += (la.topk(min(5, la.shape[1]), dim=1).indices == ya[:, None]).any(1).sum().item()
        # joint decoding, matching production: genre read off the chosen artist
        joint = la.log_softmax(1) + lg.log_softmax(1)[:, g_of_a]
        g_ok += (g_of_a[joint.argmax(1)] == yg).sum().item()
        n += ya.numel()
    model.train()
    return a_ok / n, a5_ok / n, g_ok / n


# ============================================================
# TRAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run to prove the machinery")
    ap.add_argument("--kill-test", action="store_true", help="smoke run with 15s checkpoints")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()

    smoke = a.smoke or a.kill_test
    epochs = a.epochs or (3 if smoke else EPOCHS)
    ckpt_every = 15 if a.kill_test else CKPT_EVERY_S

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | torch {torch.__version__} | "
          f"{'SMOKE' if smoke else 'FULL'} run, {epochs} epochs, ckpt every {ckpt_every}s")

    # --kill-test needs enough steps per epoch for intra-epoch checkpoints to actually
    # fire before the epoch boundary, otherwise the drill only proves epoch-end saving.
    if a.kill_test:
        lim_a, lim_p = 30, 30
    elif smoke:
        lim_a, lim_p = 12, 12
    else:
        lim_a, lim_p = 0, 0
    tr_df, va_df, artists, genres, a_idx, g_idx = build_frames(lim_a, lim_p)
    taxonomy = json.load(open(TAXONOMY, encoding="utf-8"))
    g_of_a = torch.tensor([g_idx[taxonomy[x]["genre"]] for x in artists], device=device)

    print(f"train {len(tr_df):,} / held-out {len(va_df):,} tracks | "
          f"{len(artists)} artists, {len(genres)} genres")

    tl = DataLoader(TrackAudio(tr_df, TRAIN_SECS, True), batch_size=BATCH, shuffle=True,
                    num_workers=WORKERS, drop_last=True, persistent_workers=WORKERS > 0)
    vl = DataLoader(TrackAudio(va_df, EVAL_SECS, False), batch_size=BATCH,
                    num_workers=WORKERS)

    model = PannsClassifier(len(artists), len(genres),
                            os.path.expanduser("~/panns_data/Cnn14_mAP=0.431.pth")).to(device)
    opt = torch.optim.AdamW([
        {"params": model.trunk.parameters(), "lr": LR_ENCODER},
        {"params": list(model.artist.parameters()) + list(model.genre.parameters()),
         "lr": LR_HEAD},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * max(1, len(tl)))
    # bfloat16, NOT float16. Cnn14's frontend computes a power spectrogram whose values
    # reach ~90,000 on real music, and fp16 tops out at 65,504 — it overflows to inf,
    # the log turns that into NaN, and the NaN then poisons the trunk's BatchNorm running
    # statistics permanently, so even a later fp32 eval pass returns NaN. bf16 carries
    # fp32's exponent range, so it cannot overflow here, and measured slightly FASTER
    # than fp32 (128 vs 137 ms/batch). No GradScaler is needed with bf16.
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    if device == "cuda" and not use_bf16:
        print("⚠️  bf16 desteklenmiyor — fp32'ye dusuluyor (fp16 KULLANMA, NaN verir)")
    scaler = None
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    start_epoch, step, best = 0, 0, -1.0
    if a.resume and os.path.exists(CKPT_PATH):
        start_epoch, step, best = load_checkpoint(CKPT_PATH, model, opt, sched, scaler)
        print(f"⏩ RESUME: epoch {start_epoch}, step {step:,}, best {best:.4f}")
        print(f"   LR now {opt.param_groups[0]['lr']:.2e} (encoder) — schedule position kept")
    elif a.resume:
        print("ℹ️  --resume given but no checkpoint; starting fresh")

    thermal_guard.start_monitor(TEMP_LOG)
    last_ckpt, fails = time.time(), 0
    aborted = False

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            t0, seen, run_loss = time.time(), 0, 0.0
            for x, ya, yg in tl:
                if not thermal_guard.check_and_wait():
                    print(f"\n⛔ TERMAL IPTAL: {thermal_guard.state['reason']}")
                    aborted = True
                    break

                x, ya, yg = x.to(device, non_blocking=True), ya.to(device), yg.to(device)
                opt.zero_grad(set_to_none=True)
                if use_bf16:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        la, lg = model(x)
                        loss = lossf(la, ya) + 0.5 * lossf(lg, yg)
                else:
                    la, lg = model(x)
                    loss = lossf(la, ya) + 0.5 * lossf(lg, yg)

                # Tripwire. An earlier version ran 250+ steps on NaN loss, printing it in
                # a progress line and carrying on: the optimizer skipped every update and
                # accuracy sat still while the run looked healthy. A non-finite loss means
                # the weights are already unrecoverable, so stop instead of burning hours.
                if not torch.isfinite(loss):
                    print(f"\n⛔ loss non-finite at step {step:,} — training diverged.")
                    print("   Nothing after this point is recoverable; stopping.")
                    print("   Check the autocast dtype (fp16 overflows Cnn14's frontend)"
                          " and the encoder LR.")
                    aborted = True
                    break

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                sched.step()
                step += 1; seen += ya.numel(); run_loss += loss.item()

                if time.time() - last_ckpt >= ckpt_every:
                    try:
                        save_checkpoint(CKPT_PATH, model, opt, sched, epoch, step, best, scaler)
                        write_state(epoch, step, best, f"epoch {epoch+1} icinde")
                        gt = thermal_guard.state["gpu"]
                        print(f"\n  💾 checkpoint: epoch {epoch+1}, step {step:,} "
                              f"(GPU {gt}C)", flush=True)
                        fails = 0
                    except Exception as e:
                        fails += 1
                        if fails <= 3 or fails % 25 == 0:
                            print(f"\n  ⚠️ checkpoint yazilamadi ({fails}. kez): {e}")
                            print("     ⛔ KESINTI KORUMASI SU AN ETKISIZ", flush=True)
                    finally:
                        last_ckpt = time.time()   # always, or a failure retries every step

                if step % 50 == 0:
                    el = time.time() - t0
                    print(f"  e{epoch+1} step {step:,}  loss {run_loss/max(1,seen/BATCH):.3f}  "
                          f"{seen/el:.1f} trk/s  GPU {thermal_guard.state['gpu']}C", flush=True)
            if aborted:
                break

            a1, a5, gj = evaluate(model, vl, device, g_of_a, use_bf16)
            print(f"\n=== epoch {epoch+1}/{epochs} — artist top1 {a1*100:.2f}% "
                  f"top5 {a5*100:.2f}% | genre {gj*100:.2f}%  ({time.time()-t0:.0f}s)")
            print(f"    frozen-encoder baseline: 12.96% / 32.03% / 24.17%", flush=True)

            if a1 > best:
                best = a1
                save_checkpoint(BEST_PATH, model, opt, sched, epoch + 1, step, best, scaler)
                print(f"    ✅ new best ({best*100:.2f}%) saved")
            save_checkpoint(CKPT_PATH, model, opt, sched, epoch + 1, step, best, scaler)
            write_state(epoch + 1, step, best, f"epoch {epoch+1} sonu")
            last_ckpt = time.time()

    finally:
        thermal_guard.stop_monitor()
        print(f"\n🌡️  {thermal_guard.summary()}")
        if aborted:
            print("   kosu eksik kaldi — ayni komuta --resume ekleyerek devam et")
        print(f"best artist top-1: {best*100:.2f}%  (baseline 12.96%)")


if __name__ == "__main__":
    main()
