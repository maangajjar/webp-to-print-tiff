import os
import subprocess
from pathlib import Path

from PIL import Image, ImageCms, ImageFilter

# --- Input handling (upload ONE file into input/) ---
INPUT_DIR = Path("input")
SUPPORTED_EXTS = {".webp", ".jpg", ".jpeg", ".png"}

# --- Fixed print target ---
TARGET_W = 8504
TARGET_H = 17008
DPI = 360

# --- Real-ESRGAN settings ---
MODEL = "realesrgan-x4plus"
SCALE = 4
TILE = 256
THREADS = "1:2:2"

WORK_DIR = Path("work")
OUT_DIR = Path("output")


def srgb_icc_bytes() -> bytes:
    prof = ImageCms.createProfile("sRGB")
    return ImageCms.ImageCmsProfile(prof).tobytes()


def crop_to_1_to_2(im: Image.Image) -> Image.Image:
    # Target aspect is exactly 1:2 (width:height)
    w, h = im.size
    target_w = h // 2

    if w == target_w:
        return im

    if w > target_w:
        # Center-crop width
        left = (w - target_w) // 2
        return im.crop((left, 0, left + target_w, h))

    # If image is too narrow, crop height instead (rare)
    target_h = w * 2
    if target_h > h:
        return im
    top = (h - target_h) // 2
    return im.crop((0, top, w, top + target_h))


def run_realesrgan(realesrgan_bin: str, inp: Path, out: Path) -> bool:
    cmd = [
        realesrgan_bin,
        "-i", str(inp),
        "-o", str(out),
        "-n", MODEL,
        "-s", str(SCALE),
        "-t", str(TILE),
        "-j", THREADS,
        "-f", "png",
    ]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print("Real-ESRGAN failed; fallback to normal resize. Error:", e)
        return False


def pick_input_file() -> Path:
    if not INPUT_DIR.exists():
        raise FileNotFoundError("Missing folder: input/ (create it and upload one image inside)")

    candidates = []
    for p in INPUT_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError("No input image found in input/. Upload one .jpg/.jpeg/.png/.webp")

    # Use most recently modified file
    return max(candidates, key=lambda x: x.stat().st_mtime)


def main():
    Image.MAX_IMAGE_PIXELS = None  # allow very large images

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    input_path = pick_input_file()
    base_name = input_path.stem
    out_tiff_name = f"{base_name}_600x1200mm_360dpi.tiff"

    # 1) Load and crop to exact 1:2
    im = Image.open(input_path).convert("RGB")
    im_cropped = crop_to_1_to_2(im)

    cropped_path = WORK_DIR / "input_cropped.png"
    im_cropped.save(cropped_path, format="PNG", compress_level=0)

    # 2) AI upscale (4x) with Real-ESRGAN (portable ncnn-vulkan)
    realesrgan_bin = os.environ.get("REALESRGAN_BIN", "").strip()
    upscaled_path = WORK_DIR / "upscaled.png"

    used_ai = False
    if realesrgan_bin and Path(realesrgan_bin).exists():
        used_ai = run_realesrgan(realesrgan_bin, cropped_path, upscaled_path)

    # If AI failed / missing, fallback to just using cropped image
    src_for_final = Image.open(upscaled_path).convert("RGB") if used_ai else im_cropped

    # 3) Final resize to exact print pixels
    final = src_for_final.resize((TARGET_W, TARGET_H), resample=Image.Resampling.LANCZOS)

    # Mild sharpening (conservative)
    final = final.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=3))

    # 4) Save TIFF (RGB/sRGB, 360 DPI)
    out_tiff_path = OUT_DIR / out_tiff_name
    final.save(
        out_tiff_path,
        format="TIFF",
        compression="tiff_lzw",
        dpi=(DPI, DPI),
        icc_profile=srgb_icc_bytes(),
    )

    # 5) Small preview for quick check
    preview = final.resize((2126, 4252), resample=Image.Resampling.LANCZOS)  # ~1/4 scale
    preview_path = OUT_DIR / "preview.jpg"
    preview.save(preview_path, format="JPEG", quality=92, optimize=True)

    print("\nDONE")
    print("Input file:", input_path)
    print("Input size:", im.size)
    print("Cropped size:", im_cropped.size, "(should be 1:2)")
    print("AI used:", used_ai)
    print("Final TIFF:", out_tiff_path)
    print("Final pixels:", (TARGET_W, TARGET_H), "DPI:", DPI)
    print("Preview:", preview_path)


if __name__ == "__main__":
    main()
