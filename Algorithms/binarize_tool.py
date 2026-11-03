"""
binarize_tool.py
功能：讀取 Eda hosp\\normalized_images\\<資料夾名稱>\\ 內的影像，
      對 B/G/R 三個通道分別做二值化，輸出到
      Eda hosp\\image_Binarization\\<資料夾名稱>_R / _G / _B

使用範例（main.ipynb）：
    import Algorithms.binarize_tool as binarize_tool

    binarize_tool.binarize_all_folders(
        project_path,                       # Eda hosp 根目錄
        threshold="otsu",                   # 或固定數值，例如 128
    )
    # 只處理指定資料夾：
    # binarize_tool.binarize_all_folders(project_path, folder_names=["case01"], threshold=128)
"""

from pathlib import Path

import cv2
import numpy as np

import Algorithms.image_search_tool as image_search_tool
from Algorithms.image_pd_v01 import biImg_by_threshold_leq

# cv2 讀進來是 BGR，index 對應：0=B, 1=G, 2=R
CHANNELS = {"B": 0, "G": 1, "R": 2}


def imwrite(path, img):
    """支援中文路徑的寫檔（與 region_mask_tool 相同做法）"""
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))


def get_threshold(channel, threshold):
    """threshold 為數字則直接使用；為 'otsu' 則用 Otsu 自動決定。
    Otsu 只統計非 0 像素，避免 region_mask_tool 補的黑邊/黑背景影響結果。"""
    if threshold != "otsu":
        return threshold
    valid = channel[channel > 0]
    if valid.size == 0:
        return 0
    t, _ = cv2.threshold(valid.reshape(-1, 1), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t


def binarize_channel(channel, threshold):
    """二值化單一通道，回傳 0/255 的 uint8 影像（>threshold 為 255）"""
    t = get_threshold(channel, threshold)
    binary = biImg_by_threshold_leq(channel, t)   # 0 / 1
    return (binary * 255).astype(np.uint8)


def binarize_folder(src_folder, out_root, threshold="otsu"):
    """處理單一資料夾：輸出 <out_root>\\<資料夾名>_R / _G / _B"""
    src_folder = Path(src_folder)
    out_root = Path(out_root)
    name = src_folder.name

    out_dirs = {}
    for ch in CHANNELS:
        d = out_root / f"{name}_{ch}"
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[ch] = d

    images = image_search_tool.search_images(src_folder.parent, [name], recursive=False)
    for fname, img in images:
        for ch, idx in CHANNELS.items():
            binary = binarize_channel(img[:, :, idx], threshold)
            imwrite(out_dirs[ch] / fname, binary)
    print(f"✅ {name}: {len(images)} 張 → {out_root}\\{name}_R/G/B")


def binarize_all_folders(root_path, threshold="otsu", folder_names=None,
                         src_name="normalized_images", out_name="image_Binarization"):
    """批次處理 root_path\\normalized_images 下的所有（或指定）資料夾"""
    root_path = Path(root_path)
    src_root = root_path / src_name
    out_root = root_path / out_name
    if not src_root.is_dir():
        raise FileNotFoundError(f"路徑不存在: {src_root}")

    if folder_names is None:
        folders = sorted(p for p in src_root.iterdir() if p.is_dir())
    else:
        folders = [src_root / n for n in folder_names]

    if not folders:
        print(f"在 {src_root} 下沒有找到資料夾。")
        return

    for folder in folders:
        binarize_folder(folder, out_root, threshold)