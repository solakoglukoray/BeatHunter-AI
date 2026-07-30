"""
Thermal supervision for long GPU runs.

This machine thermally shut down at 96-97C during a benchmark, and a soak test
afterwards showed the GPU still climbing 4-6 C/min after three minutes at only 58W —
cooling is marginal, so any sustained load needs watching.

CPU temperature is NOT readable here: WSL2 exposes no thermal_zone, and the Windows WMI
thermal class needs admin. This guards on GPU temperature only.

Usage:
    start_monitor()                       # begins polling + logging
    callbacks=[ThermalGuard(), ...]       # inside model.fit
    stop_monitor()                        # in a finally block
    summary()                             # peak temperature, pause count
"""

import os
import time
import threading
import subprocess

ABORT_C, COOL_C, RESUME_C = 90, 80, 70
COOL_TIMEOUT = 600
POLL_SECONDS = 2

state = {"gpu": None, "max": 0, "abort": False, "reason": None, "pauses": 0}
_stop = threading.Event()
_thread = None


def read_gpu():
    """Returns (temp_c, power_w, clock_mhz), or (None, None, None) if unreadable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().split(",")
        return int(out[0]), float(out[1]), float(out[2])
    except Exception:
        return None, None, None


def _poll(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.write("t,gpu_temp_c,gpu_power_w,gpu_clock_mhz\n")
        t0 = time.time()
        while not _stop.is_set():
            t, p, c = read_gpu()
            if t is not None:
                f.write(f"{time.time()-t0:.1f},{t},{p},{c}\n")
                f.flush()
                state["gpu"] = t
                state["max"] = max(state["max"], t)
                if t >= ABORT_C:
                    state["abort"] = True
                    state["reason"] = f"GPU {t}C >= {ABORT_C}C abort esigi"
            _stop.wait(POLL_SECONDS)


def start_monitor(log_path):
    """Polls in the background so the training loop only reads a cached value."""
    global _thread
    _stop.clear()
    state.update({"gpu": None, "max": 0, "abort": False, "reason": None, "pauses": 0})
    _thread = threading.Thread(target=_poll, args=(log_path,), daemon=True)
    _thread.start()
    time.sleep(POLL_SECONDS + 0.5)          # let the first reading land
    t, _, _ = read_gpu()
    print(f"🌡️  termal guard: baslangic {t}C | "
          f"{COOL_C}C durakla / {RESUME_C}C devam / {ABORT_C}C iptal")


def stop_monitor():
    _stop.set()
    if _thread:
        _thread.join(timeout=5)


def summary():
    return f"GPU tepe {state['max']}C, {state['pauses']} termal duraklama"


class ThermalGuard:
    """
    Keras callback base. Import as a mixin target — see make_keras_callback().

    Checked every `every` batches and at each epoch end. Reads the cached temperature
    rather than shelling out, so the per-batch cost is a dict lookup. Pausing blocks
    inside the callback, which idles the GPU without losing epoch state.
    """

    def __init__(self, every=20):
        self.every = every

    def _check(self, model):
        if state["abort"]:
            model.stop_training = True
            return
        t = state["gpu"]
        if t is None or t < COOL_C:
            return

        state["pauses"] += 1
        print(f"\n  ~ GPU {t}C >= {COOL_C}C — duraklatildi, <{RESUME_C}C bekleniyor...",
              flush=True)
        waited = 0
        while True:
            time.sleep(5)
            waited += 5
            t = state["gpu"]
            if t is None or t < RESUME_C:
                print(f"  ~ devam ({t}C, {waited}s duraklama)", flush=True)
                return
            if t >= ABORT_C or waited > COOL_TIMEOUT:
                state["abort"] = True
                state["reason"] = f"soguma basarisiz (GPU {t}C, {waited}s)"
                model.stop_training = True
                return


def check_and_wait():
    """
    Framework-agnostic guard for a plain training loop (PyTorch, or anything else).

    Returns True to keep going, False if the run must stop. Blocks while the GPU cools,
    which idles it without losing any state the caller holds.

    The Keras path below does the same thing through a callback because Keras owns the
    loop; here the caller owns it, so this just returns a verdict.
    """
    if state["abort"]:
        return False
    t = state["gpu"]
    if t is None or t < COOL_C:
        return True

    state["pauses"] += 1
    print(f"\n  ~ GPU {t}C >= {COOL_C}C — duraklatildi, <{RESUME_C}C bekleniyor...",
          flush=True)
    waited = 0
    while True:
        time.sleep(5)
        waited += 5
        t = state["gpu"]
        if t is None or t < RESUME_C:
            print(f"  ~ devam ({t}C, {waited}s duraklama)", flush=True)
            return True
        if t >= ABORT_C or waited > COOL_TIMEOUT:
            state["abort"] = True
            state["reason"] = f"soguma basarisiz (GPU {t}C, {waited}s)"
            return False


def make_keras_callback(tf, every=20):
    """Builds a tf.keras.callbacks.Callback bound to the guard (tf passed in to avoid
    importing TensorFlow from this module)."""

    guard = ThermalGuard(every=every)

    class _Cb(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            if batch % guard.every == 0:
                guard._check(self.model)

        def on_epoch_end(self, epoch, logs=None):
            guard._check(self.model)

    return _Cb()
