script_path = "/content/drive/MyDrive/openvoicedata/run_convert_debug.py"

code = r'''
import os
from pathlib import Path

# ===== hạn chế crash OpenMP =====
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ép CPU để test ổn định trước

DATA_DIR = Path("/content/drive/MyDrive/openvoicedata")
SRC_WAV  = DATA_DIR / "source.wav"
REF_WAV  = DATA_DIR / "ref.wav"
OUT_WAV  = DATA_DIR / "out_converted.wav"

print("SRC:", SRC_WAV, SRC_WAV.exists(), flush=True)
print("REF:", REF_WAV, REF_WAV.exists(), flush=True)
print("OUT:", OUT_WAV, flush=True)

CKPT_DIR = Path("/content/drive/MyDrive/checkpoints_v2")
print("CKPT_DIR:", CKPT_DIR, CKPT_DIR.exists(), flush=True)

cfg = CKPT_DIR / "converter" / "config.json"
ckpt = CKPT_DIR / "converter" / "checkpoint.pth"
if not cfg.exists() or not ckpt.exists():
    cfg_list = list(CKPT_DIR.rglob("config.json"))
    ckpt_list = list(CKPT_DIR.rglob("checkpoint.pth"))
    if cfg_list: cfg = cfg_list[0]
    if ckpt_list: ckpt = ckpt_list[0]
print("config:", cfg, flush=True)
print("ckpt  :", ckpt, flush=True)

device = "cpu"
print("device:", device, flush=True)

print("Importing openvoice...", flush=True)
from openvoice.api import ToneColorConverter
from openvoice import se_extractor
print("✅ imported", flush=True)

print("Loading converter...", flush=True)
converter = ToneColorConverter(str(cfg), device=device)
converter.load_ckpt(str(ckpt))
print("✅ converter loaded", flush=True)

print("Extracting embeddings...", flush=True)
src_se, _ = se_extractor.get_se(str(SRC_WAV), converter, vad=False)
tgt_se, _ = se_extractor.get_se(str(REF_WAV), converter, vad=False)
print("✅ embeddings ok", flush=True)

print("Converting...", flush=True)
converter.convert(
    audio_src_path=str(SRC_WAV),
    src_se=src_se,
    tgt_se=tgt_se,
    output_path=str(OUT_WAV),
    message="convert"
)

print("🎉 DONE:", OUT_WAV, "size:", OUT_WAV.stat().st_size, "bytes", flush=True)
'''

with open(script_path, "w", encoding="utf-8") as f:
    f.write(code)

print("Wrote:", script_path)
