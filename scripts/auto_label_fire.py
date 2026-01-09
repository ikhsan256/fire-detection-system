from ultralytics import YOLO
from pathlib import Path
import cv2
import shutil
import random
from tqdm import tqdm

# 1. Path dataset asli
SRC = Path("dataset_path")

# 2. Output YOLO dataset
OUT = Path("data/fire_yolo")

model = YOLO("yolov8n.pt")

# 3. Buat folder YOLO
for p in [
    OUT/"images/train",
    OUT/"images/val",
    OUT/"labels/train",
    OUT/"labels/val"
]:
    p.mkdir(parents=True, exist_ok=True)

def split():
    return "val" if random.random() < 0.2 else "train"

# 4. FIRE images → auto bbox
fire_imgs = list((SRC/"Fire").glob("*.*"))

for img_path in tqdm(fire_imgs):
    split_set = split()
    img = cv2.imread(str(img_path))
    h, w, _ = img.shape

    results = model(img, conf=0.4)[0]

    shutil.copy(img_path, OUT/f"images/{split_set}/{img_path.name}")

    label_file = OUT/f"labels/{split_set}/{img_path.stem}.txt"

    with open(label_file, "w") as f:
        for box in results.boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            xc = ((x1+x2)/2)/w
            yc = ((y1+y2)/2)/h
            bw = (x2-x1)/w
            bh = (y2-y1)/h
            f.write(f"0 {xc} {yc} {bw} {bh}\n")

# 5. NON-FIRE images → empty label
nonfire_imgs = list((SRC/"Non_Fire").glob("*.*"))

for img_path in tqdm(nonfire_imgs):
    split_set = split()
    shutil.copy(img_path, OUT/f"images/{split_set}/{img_path.name}")
    open(OUT/f"labels/{split_set}/{img_path.stem}.txt", "w").close()

print("Auto-labeling selesai")



