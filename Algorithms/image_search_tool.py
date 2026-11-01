"""
image_search_tool.py
功能：由根目錄 + 逐層資料夾名稱組出路徑，收集資料夾內的圖片，並轉為 cv2 格式。

使用範例（main.ipynb）：
    import Algorithms.image_search_tool as image_search_tool

    PATH_LEVELS = ["image study", "test"]
    RECURSIVE = False

    images = image_search_tool.search_images(project_path, PATH_LEVELS, RECURSIVE)
    image_search_tool.show_images_grid(images, cols=6)
"""

import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def build_target_path(root_path, path_levels):
    """根據 path_levels 由 root_path 逐層組出目標資料夾路徑"""
    target = Path(root_path)
    for level in path_levels:
        target = target / level
    if not target.exists():
        raise FileNotFoundError(f"路徑不存在: {target}")
    if not target.is_dir():
        raise NotADirectoryError(f"不是資料夾: {target}")
    return target


def collect_images_in_folder(folder_path, recursive=False):
    """收集資料夾內所有圖片檔案路徑（回傳排序後的 Path list）"""
    folder_path = Path(folder_path)
    pattern = "**/*" if recursive else "*"
    image_files = sorted(
        p for p in folder_path.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"在 '{folder_path}' 中找到 {len(image_files)} 張圖片。")
    return image_files


def load_images(image_paths):
    """用 cv2 將圖片讀入記憶體，回傳 [(檔名, 圖片陣列), ...]

    使用 np.fromfile + cv2.imdecode，避免 cv2.imread 在 Windows 上
    遇到中文路徑讀不到的問題。
    """
    images = []
    for p in image_paths:
        try:
            data = np.fromfile(str(p), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except OSError:
            img = None
        if img is None:
            print(f"⚠️ 無法讀取: {p}")
            continue
        images.append((p.name, img))
    return images


def search_images(root_path, path_levels, recursive=False):
    """一次完成：組路徑 → 收集圖片路徑 → 讀成 cv2 格式

    參數:
        root_path:   根目錄（main.ipynb 所在位置）
        path_levels: 由根目錄開始的逐層資料夾名稱 list
        recursive:   是否連同子資料夾內的圖片一起收集

    回傳:
        [(檔名, cv2 圖片陣列), ...]
    """
    target_folder = build_target_path(root_path, path_levels)
    print(f"目標資料夾: {target_folder}")

    image_paths = collect_images_in_folder(target_folder, recursive=recursive)
    return load_images(image_paths)


def show_images_grid(images, cols=4, figsize_per_img=(3, 3)):
    """用網格方式顯示所有圖片

    參數:
        images:          [(檔名, cv2 BGR 圖片陣列), ...]
        cols:            每列顯示幾張
        figsize_per_img: 每張圖的顯示大小 (寬, 高)
    """
    n = len(images)
    if n == 0:
        print("沒有圖片可以顯示。")
        return

    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(figsize_per_img[0] * cols, figsize_per_img[1] * rows)
    )

    # 用子圖總數攤平，不論 axes 原本是單一物件或 1D/2D array 都能處理
    axes = np.array(axes).reshape(-1)

    for ax, (name, img) in zip(axes, images):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # cv2 是 BGR，顯示前轉 RGB
        ax.imshow(img_rgb)
        ax.set_title(name, fontsize=8)
        ax.axis('off')

    # 把多餘的空格關掉
    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()