import os
import subprocess
from pathlib import Path

from PIL import Image, ImageCms, ImageFilter

INPUT_WEBP = "HL03-HolosAndamanEmerald-Minimale.webp"

TARGET_W = 8504
TARGET_H = 17008
DPI = 360

MODEL = "realesrgan-x4plus"
SCALE = 4
TILE = 256
THREADS = "1:2:2"

OUT_TIFF = "HL03-HolosAndamanEmerald-Minimale_600x1200mm_360dpi.tiff"

WORK_DIR = Path("work")
OUT_DIR = Path("output")


def srgb_icc_bytes() -> bytes:
    prof = ImageCms.createProfile("sRGB")
    return ImageCms.ImageCmsProfile(prof).tobytes()


def crop_to_1_to_2(im: Image.Image) -> Image.Image:
    w, h = im.size
    target_w = h // 2
    if w == target_w:
        return im
    if w > target_w:
        left = (w - target_w) // 2
        return im.crop((left, 0, left + target_w, h))
    # too narrow: crop height
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


def main():
    Image.MAX_IMAGE_PIXELS = None

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    input_path = Path(INPUT_WEBP)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_WEBP}")

    im = Image.open(input_path).convert("RGB")
    im_cropped = crop_to_1_to_2(im)

    cropped_path = WORK_DIR / "input_cropped.png"
    im_cropped.save(cropped_path, format="PNG", compress_level=0)

    realesrgan_bin = os.environ.get("REALESRGAN_BIN", "").strip()
    upscaled_path = WORK_DIR / "upscaled.png"

    used_ai = False
    if realesrgan_bin and Path(realesrgan_bin).exists():
        used_ai = run_realesrgan(realesrgan_bin, cropped_path, upscaled_path)

    src = Image.open(upscaled_path).convert("RGB") if used_ai else im_cropped

    final = src.resize((TARGET_W, TARGET_H), resample=Image.Resampling.LANCZOS)
    final = final.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=3))

    out_tiff_path = OUT_DIR / OUT_TIFF
    final.save(
        out_tiff_path,
        format="TIFF",
        compression="tiff_lzw",
        dpi=(DPI, DPI),
        icc_profile=srgb_icc_bytes(),
    )

    preview = final.resize((2126, 4252), resample=Image.Resampling.LANCZOS)
    preview.save(OUT_DIR / "preview.jpg", format="JPEG", quality=92, optimize=True)

    print("DONE")
    print("Input:", im.size)
    print("Cropped:", im_cropped.size)
    print("AI used:", used_ai)
    print("Output:", out_tiff_path, "pixels:", (TARGET_W, TARGET_H), "dpi:", DPI)


if __name__ == "__main__":
    main()
