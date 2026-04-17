from pathlib import Path
import cv2
from tqdm import tqdm

from .add_single_degradation import *


def degrade(img, degradation, idx):
    router = {
        "low resolution": lr,
        "dark": darken,
        "noise": add_noise,
        "jpeg compression artifact": add_jpeg_comp_artifacts,
        "haze": add_haze,
        "motion blur": add_motion_blur,
        "defocus blur": add_defocus_blur,
        "rain": add_rain,
    }
    if degradation == "haze":
        return add_haze(img, idx=idx)
    return router[degradation](img)


base_dir = Path("dataset")
hq_dir = base_dir / "HQ"
degras_path = base_dir / "degradations.txt"

lq_dir = base_dir / "LQ"
lq_dir.mkdir(exist_ok=True)

with open(degras_path) as f:
    lines = f.readlines()
combs = []
for line in lines:
    items = [i.strip() for i in line.strip().split('+')]
    degras = [i for i in items if i]
    if degras:
        combs.append(degras)

hq_paths = sorted(list(hq_dir.glob("*")))
for comb in combs:
    n_degra = len(comb)
    comb_dir = lq_dir / f"d{n_degra}" / "+".join(comb)
    comb_dir.mkdir(parents=True)
    for hq_path in tqdm(hq_paths, desc=" + ".join(comb), unit='img'):
        img = cv2.imread(str(hq_path))
        for degra in comb:
            img = degrade(img, degra, idx=hq_path.stem)
        cv2.imwrite(str(comb_dir / hq_path.name), img)