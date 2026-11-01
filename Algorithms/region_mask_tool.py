r"""
互動式「正方形範圍 + 容許誤差區域擴張」遮罩工具（支援資料夾批次、Jupyter）

【Jupyter】
    import region_mask_tool
    region_mask_tool.run(r"c:\Users\tntki\OneDrive\Desktop\Eda Hosp\image study\test", 200)
    #                     資料夾/圖片路徑                                                      正方形邊長 n(預設 200)
    # 進階參數: run(路徑, n, crop=True, out_dir=None, use_tolerance=True, use_blur=True)
    #   crop=True  輸出裁切成 n×n 的圖（所有圖大小一致 = 大小規格化）
    #   crop=False 輸出維持原圖大小（範圍外全黑）
    #   out_dir    輸出位置，預設為目前工作目錄（通常就是 main.ipynb 所在資料夾）
    #   use_tolerance=False  關閉 Tolerance：不做區域擴張，正方形內全部保留（Blur 也會一併失效）
    #   use_blur=False       關閉 Blur：判斷連接時不先模糊（Tolerance 仍可用）

【命令列】
    python region_mask_tool.py "資料夾或圖片路徑" [n]

處理流程（點擊一個點後）：
  1. 只保留以該點為中心、邊長 n 的正方形內的影像，其餘全黑
  2. 在正方形內，從該點出發做「容許誤差的洪水填充」：
     與基準灰階差距在 Tolerance 內且互相連通的像素才保留，其餘全黑

操作:
  左鍵單擊               以該點為中心放置正方形（沿用目前邊長）
  左鍵按下→拖曳→放開     按下處=中心，拖到哪裡=正方形的角（直接在圖上決定大小）
  右鍵拖曳               只調整正方形大小（中心不動）
  滑鼠滾輪               微調大小（部分系統支援）
  滑桿   Size(正方形邊長) / Tolerance(連接容許誤差) / Blur(先模糊抗雜訊)
  f      切換：只做步驟1(僅正方形) <-> 步驟1+2
  m      切換比較基準：fixed(與中心點比) <-> floating(與相鄰像素比，可跨漸層)
  n / p  下一張 / 上一張（每張圖各自記住自己的選取與參數，一張圖只記最後一次選取）
  s      全部選完後按：在 out_dir 建立資料夾，把「有選取範圍的圖」全部輸出（檔名沿用原檔名）
  r      取消目前這張圖的選取      q 或 Esc 離開
需求: pip install opencv-python numpy
"""
import sys
import os
import glob
import cv2
import numpy as np

WIN = "Region Tool"
MAX_DISPLAY_W = 700
DEFAULT_N = 200
EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


def imread_gray(path):
    data = np.fromfile(path, dtype=np.uint8)  # 支援中文/空格路徑
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"無法讀取圖片: {path}")
    return img


def imwrite(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)


def collect_images(target):
    if os.path.isdir(target):
        files = []
        for e in EXTS:
            files += glob.glob(os.path.join(target, e))
            files += glob.glob(os.path.join(target, e.upper()))
        return sorted(set(files))
    if os.path.isfile(target):
        return [target]
    raise FileNotFoundError(f"找不到路徑: {target}")


def square_corners(seed, size):
    """以 seed 為中心、邊長 size 的正方形左上/右下角座標（可超出圖片邊界）"""
    size = max(1, int(size))
    half = size // 2
    x0, y0 = seed[0] - half, seed[1] - half
    return (x0, y0), (x0 + size - 1, y0 + size - 1)


def segment(gray, seed, size, tol, blur, fixed=True, use_flood=True):
    """回傳 (region_mask 0/255, square_mask 0/255)"""
    h, w = gray.shape
    square = np.zeros((h, w), np.uint8)
    p1, p2 = square_corners(seed, size)
    cv2.rectangle(square, p1, p2, 255, -1)  # OpenCV 會自動裁切超出邊界的部分
    if not use_flood:
        return square.copy(), square

    src = gray
    if blur > 0:
        k = 2 * int(blur) + 1
        src = cv2.GaussianBlur(gray, (k, k), 0)

    # floodFill 的 mask：非 0 = 禁止填入。正方形外全部設為禁止，所以不會漏出範圍外
    mask = np.zeros((h + 2, w + 2), np.uint8)
    mask[1:-1, 1:-1] = (square == 0).astype(np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    if fixed:
        flags |= cv2.FLOODFILL_FIXED_RANGE
    t = int(tol)
    cv2.floodFill(src.copy(), mask, seed, 0, (t, t, t, t), (t, t, t, t), flags)
    region = (mask[1:-1, 1:-1] == 255).astype(np.uint8) * 255
    return region, square


def render_selection(gray, sel, crop=True):
    """依單張圖的選取參數產生輸出圖。crop=True -> n×n 裁切（超出圖片的部分補黑）"""
    region, _ = segment(gray, sel["seed"], sel["size"], sel["tol"], sel["blur"],
                        sel["fixed"], sel["flood"])
    masked = cv2.bitwise_and(gray, gray, mask=region)
    if not crop:
        return masked
    size = int(sel["size"])
    (x0, y0), _ = square_corners(sel["seed"], size)
    out = np.zeros((size, size), np.uint8)
    h, w = gray.shape
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + size, w), min(y0 + size, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = masked[sy0:sy1, sx0:sx1]
    return out


class RegionTool:
    def __init__(self, files, n=DEFAULT_N, crop=True, out_dir=None,
                 use_tolerance=True, use_blur=True):
        self.files = files
        self.use_tolerance = bool(use_tolerance)
        self.use_blur = bool(use_blur) and self.use_tolerance  # 沒有 Tolerance 就不需要 Blur
        self.n = max(1, int(n))
        self.crop = crop
        self.out_dir = out_dir or os.getcwd()
        self.sel = {}          # 圖片路徑 -> 選取參數
        self.save_dir = None   # 本次執行的輸出資料夾（第一次存檔時決定，之後重複覆蓋同一個）
        self.seed = None
        self.region = self.square = self.result = None
        self.fixed = True
        self.use_flood = self.use_tolerance
        self.message = ""
        self._lock = False
        self._drag_start = None
        self._drag_mode = None

        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Size", WIN, min(self.n, 2000), max(2000, self.n), self.refresh)
        if self.use_tolerance:
            cv2.createTrackbar("Tolerance", WIN, 20, 100, self.refresh)
        if self.use_blur:
            cv2.createTrackbar("Blur", WIN, 2, 10, self.refresh)
        cv2.setMouseCallback(WIN, self.on_mouse)
        self.load(0)

    def load(self, idx):
        self.idx = idx % len(self.files)
        self.path = self.files[self.idx]
        self.gray = imread_gray(self.path)
        self.h, self.w = self.gray.shape
        self.scale = min(1.0, MAX_DISPLAY_W / self.w)
        self.region = self.square = self.result = None
        try:
            # 滑桿上限 = 圖片較長邊，但不低於目前的大小，避免被截斷
            cv2.setTrackbarMax("Size", WIN, max(self.h, self.w, self.cur_size()))
        except Exception:
            pass
        sel = self.sel.get(self.path)
        if sel:  # 這張圖之前選過 -> 還原
            self.seed = sel["seed"]
            self.fixed, self.use_flood = sel["fixed"], sel["flood"]
            self._lock = True
            cv2.setTrackbarPos("Size", WIN, int(sel["size"]))
            if self.use_tolerance:
                cv2.setTrackbarPos("Tolerance", WIN, int(sel["tol"]))
            if self.use_blur:
                cv2.setTrackbarPos("Blur", WIN, int(sel["blur"]))
            self._lock = False
        else:
            self.seed = None
        self.message = "click: place square | drag: set size | s: save all"
        self.refresh()

    def cur_size(self):
        return max(1, cv2.getTrackbarPos("Size", WIN))

    def set_size(self, s):
        s = int(min(max(s, 1), max(self.h, self.w)))
        self._lock = True
        cv2.setTrackbarPos("Size", WIN, s)  # 同步滑桿
        self._lock = False

    def compute(self):
        if self.seed is None:
            self.region = self.square = self.result = None
            return
        size = self.cur_size()
        tol = cv2.getTrackbarPos("Tolerance", WIN) if self.use_tolerance else 0
        blur = cv2.getTrackbarPos("Blur", WIN) if self.use_blur else 0
        self.region, self.square = segment(
            self.gray, self.seed, size, tol, blur, self.fixed, self.use_flood)
        self.result = cv2.bitwise_and(self.gray, self.gray, mask=self.region)
        self.sel[self.path] = dict(seed=self.seed, size=size, tol=tol, blur=blur,
                                   fixed=self.fixed, flood=self.use_flood)
        mode = "fixed" if self.fixed else "floating"
        step = f"square+flood({mode})" if self.use_flood else "square only"
        params = f"n={size}"
        if self.use_tolerance:
            params += f" tol={tol}"
        if self.use_blur:
            params += f" blur={blur}"
        self.message = (f"[{self.idx + 1}/{len(self.files)}] {os.path.basename(self.path)} "
                        f"| {step} | {params} "
                        f"| area={int(np.count_nonzero(self.region))}px | selected={len(self.sel)}")

    def outlined(self):
        color = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
        if self.square is not None:
            p1, p2 = square_corners(self.seed, self.cur_size())
            cv2.rectangle(color, p1, p2, (0, 255, 255), 1)  # 黃色=正方形範圍
        if self.region is not None:
            cnts, _ = cv2.findContours(self.region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(color, cnts, -1, (0, 0, 255), 2)  # 紅色=保留區域
        if self.seed is not None:
            cv2.circle(color, self.seed, 4, (0, 255, 0), -1)
        return color

    def refresh(self, *_):
        if not hasattr(self, "gray") or self._lock:
            return
        self.compute()
        left = self.outlined()
        right = cv2.cvtColor(
            self.result if self.result is not None else np.zeros_like(self.gray),
            cv2.COLOR_GRAY2BGR)
        if self.scale < 1.0:
            size = (int(self.w * self.scale), int(self.h * self.scale))
            left = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, size, interpolation=cv2.INTER_AREA)
        canvas = np.hstack([left, right])
        cv2.putText(canvas, self.message, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(WIN, canvas)

    @staticmethod
    def side_from_drag(cx, cy, x, y):
        # 正方形以中心對稱，拖到的點落在其角/邊上 -> 邊長 = 2 * 最大偏移量
        return 2 * max(abs(x - cx), abs(y - cy))

    def on_mouse(self, event, mx, my, flags, param):
        if not hasattr(self, "gray"):
            return
        x = min(max(int(mx / self.scale), 0), self.w - 1)
        y = min(max(int(my / self.scale), 0), self.h - 1)
        in_left = mx < int(self.w * self.scale)

        if event == cv2.EVENT_LBUTTONDOWN and in_left:
            self._drag_mode = "square"
            self._drag_start = (x, y)
            self.seed = (x, y)
            self.refresh()
        elif event == cv2.EVENT_RBUTTONDOWN and in_left and self.seed is not None:
            self._drag_mode = "size"
        elif event == cv2.EVENT_MOUSEMOVE and self._drag_mode:
            if self._drag_mode == "square":
                sx, sy = self._drag_start
                if max(abs(x - sx), abs(y - sy)) >= 3:  # 有實際拖曳才改大小
                    self.set_size(self.side_from_drag(sx, sy, x, y))
            elif self._drag_mode == "size" and self.seed is not None:
                self.set_size(self.side_from_drag(self.seed[0], self.seed[1], x, y))
            self.refresh()
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self._drag_mode = None
        elif event == cv2.EVENT_MOUSEWHEEL and self.seed is not None:
            cur = self.cur_size()
            step = max(2, int(cur * 0.08))
            self.set_size(cur + (step if flags > 0 else -step))
            self.refresh()

    def folder_name(self):
        # 格式: normalized_<圖片上一層資料夾名稱>_<n>x<n>
        parent = os.path.basename(os.path.dirname(os.path.abspath(self.files[0])))
        parent = "".join("_" if c in '\\/:*?"<>|' else c for c in parent).strip() or "images"
        sizes = sorted({int(v["size"]) for v in self.sel.values()})
        spec = f"{sizes[0]}x{sizes[0]}" if len(sizes) == 1 else "mixed_size"
        prefix = "normalized" if self.crop else "masked"  # crop=False 時圖片並未規格化，改用 masked 前綴
        return f"{prefix}_{parent}_{spec}"

    def save(self):
        if not self.sel:
            self.message = "Nothing selected yet - click on an image first"
            self.refresh()
            return
        if self.save_dir is None:  # 第一次存檔：建立不重名的資料夾
            base = os.path.join(self.out_dir, self.folder_name())
            folder, k = base, 2
            while os.path.exists(folder):
                folder, k = f"{base}_{k}", k + 1
            os.makedirs(folder)
            self.save_dir = folder
        count = 0
        for path in self.files:  # 依檔名順序輸出，名稱沿用
            if path in self.sel:
                gray = self.gray if path == self.path else imread_gray(path)
                out = render_selection(gray, self.sel[path], self.crop)
                imwrite(os.path.join(self.save_dir, os.path.basename(path)), out)
                count += 1
        print(f"已輸出 {count} 張圖片到: {self.save_dir}")
        self.message = f"Saved {count} images -> {os.path.basename(self.save_dir)}"
        self.refresh()

    def loop(self):
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("n"):
                self.load(self.idx + 1)
            elif key == ord("p"):
                self.load(self.idx - 1)
            elif key == ord("r"):
                self.sel.pop(self.path, None)
                self.seed = None
                self.message = "reset - click a point"
                self.refresh()
            elif key == ord("s"):
                self.save()
            elif key == ord("f") and self.use_tolerance:
                self.use_flood = not self.use_flood
                self.refresh()
            elif key == ord("m") and self.use_tolerance:
                self.fixed = not self.fixed
                self.refresh()
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)


def run(target, n=DEFAULT_N, crop=True, out_dir=None, use_tolerance=True, use_blur=True):
    """target: 資料夾或單張圖片路徑；n: 正方形邊長預設值(像素)
    crop: True=輸出裁成 n×n；False=保持原圖大小。out_dir: 輸出位置，預設目前工作目錄
    use_tolerance: False=關閉容許誤差區域擴張（正方形內全保留）
    use_blur: False=關閉判斷前的模糊"""
    files = collect_images(target)
    if not files:
        raise FileNotFoundError(f"在 {target} 找不到圖片")
    print(f"找到 {len(files)} 張圖片，正方形邊長 n={n}。"
          "單擊放置正方形，拖曳可調整大小；n/p 換圖；全部選完按 s 一次輸出；q 離開")
    RegionTool(files, n, crop, out_dir, use_tolerance, use_blur).loop()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--f=")]  # 過濾 Jupyter 參數
    paths = [a for a in args if os.path.exists(a)]
    if not paths:
        print('用法: python region_mask_tool.py "資料夾或圖片路徑" [n]')
        print("Jupyter: import region_mask_tool; region_mask_tool.run(r'你的資料夾', 200)")
    else:
        nums = [a for a in args if a.isdigit()]
        run(paths[0], int(nums[0]) if nums else DEFAULT_N)