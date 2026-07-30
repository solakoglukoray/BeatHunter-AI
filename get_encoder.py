"""
Encoder weight bootstrap — run this on a fresh machine before anything else.

WHY THIS IS NOT A DOWNLOAD SCRIPT (for the fine-tuned encoder)
──────────────────────────────────────────────────────────────
There are two weight files and they are fundamentally different kinds of thing:

  models/panns_ft_encoder.pt   327 MB   OUR fine-tuned encoder. Not published anywhere.
                                        It is a BUILD ARTEFACT of panns_finetune.py.
  ~/panns_data/Cnn14_...pth    327 MB   Stock AudioSet weights. Public, downloadable.

So "just download it" only works for the second one. For the first there are three
honest options, in order of what you probably want:

  1. COPY it from a machine that already has it (fastest, and what you want day to day).
     Put it at models/panns_ft_encoder.pt. This script verifies it afterwards.

  2. REGENERATE it — ~2.8 hours on an RTX 4080:
         python panns_finetune.py
         python panns_ft_frozen_experiment.py --extract-encoder
     Then rebuild the classifier and catalogue:
         python train_classic.py --extract          (~45 min, re-embeds every track)
     Note the result will not be bit-identical, so the model bundle's encoder hash will
     differ from the committed one. That is expected; the rebuild writes a matching pair.

  3. RUN WITHOUT IT. Delete or move models/panns_ft_encoder.pt and the stock AudioSet
     weights take over automatically, at a measured cost:
         fine-tuned  artist top-1 15.35%  top-5 37.72%  genre 28.21%
         stock       artist top-1 12.96%  top-5 32.03%  genre 24.17%
     You must then rebuild: python train_classic.py  (minutes, embeddings are cached).
     inference_engine refuses to mix a catalogue built by one encoder with the other.

Git deliberately does not track either file. 327 MB of binary that changes whenever the
model is retrained is what LFS quotas are for, and the repo has no LFS set up. If that
changes, track models/panns_ft_encoder.pt and leave the cached embeddings out.

    python get_encoder.py            # check what is present, fetch the stock weights
    python get_encoder.py --verify   # check only, change nothing
"""

import argparse
import hashlib
import os
import sys
import urllib.request

FT_CKPT   = "models/panns_ft_encoder.pt"
MODEL_BUNDLE = "models/classic_model.joblib"
PANNS_DIR = os.path.expanduser("~/panns_data")
BASE_CKPT = os.path.join(PANNS_DIR, "Cnn14_mAP=0.431.pth")
LABELS    = os.path.join(PANNS_DIR, "class_labels_indices.csv")

BASE_URL   = ("https://zenodo.org/record/3987831/files/"
              "Cnn14_mAP%3D0.431.pth?download=1")
LABELS_URL = ("http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/"
              "class_labels_indices.csv")


def short_hash(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def download(url, dest, label):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"downloading {label} -> {dest}")

    def hook(blocks, bs, total):
        if total > 0:
            pct = min(100, blocks * bs * 100 // total)
            print(f"\r  {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print(f"\r  done ({os.path.getsize(dest)/1e6:.0f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="report only, download nothing")
    args = ap.parse_args()

    print("=== encoder weights ===")
    ok = True

    if os.path.exists(FT_CKPT):
        h = short_hash(FT_CKPT)
        print(f"  fine-tuned : present  {os.path.getsize(FT_CKPT)/1e6:.0f} MB  hash {h}")
        if os.path.exists(MODEL_BUNDLE):
            import joblib
            want = joblib.load(MODEL_BUNDLE).get("encoder_hash")
            if want and want != h:
                print(f"    ⚠️  model bundle expects hash {want} — MISMATCH.")
                print("       inference_engine will refuse to run. Rebuild with:")
                print("         python train_classic.py")
                ok = False
            elif want:
                print(f"    matches the model bundle ✓")
    else:
        print(f"  fine-tuned : MISSING ({FT_CKPT})")
        print("    Not downloadable — it is a build artefact. Copy it from another")
        print("    machine, or regenerate (see this file's docstring). Without it the")
        print("    stock AudioSet weights are used instead, costing ~2.4 points of")
        print("    artist top-1 (15.35% -> 12.96%), and you must run train_classic.py.")

    for path, url, label in ((BASE_CKPT, BASE_URL, "stock AudioSet Cnn14 (327 MB)"),
                             (LABELS, LABELS_URL, "AudioSet class labels")):
        if os.path.exists(path):
            print(f"  {os.path.basename(path)[:26]:26s}: present")
        elif args.verify:
            print(f"  {os.path.basename(path)[:26]:26s}: MISSING")
            ok = False
        else:
            try:
                download(url, path, label)
            except Exception as e:
                print(f"  failed: {e}")
                ok = False

    print("\nOK" if ok else "\nsomething needs attention (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
