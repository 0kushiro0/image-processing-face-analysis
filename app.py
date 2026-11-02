"""AI 輔助之互動式影像處理與人臉分析工具 (Gradio)

模組:
  1. 影像載入與基礎解析
  2. 色彩空間轉換 (RGB / YUV) 與同步直方圖
  3. 自訂卷積核空間濾波 + PSNR
  4. 人臉偵測 (YuNet 輕量深度學習偵測器, 失敗時退回 Haar-Cascade)
"""
import os

import cv2
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
MAX_SIDE = 1600          # 超過則等比縮小, 避免卷積過慢
MAX_KERNEL = 15


# ---------------------------------------------------------------- 模組 1
def load_image(path):
    """讀取影像 -> (RGB uint8 陣列, 說明文字)。清除上傳時回傳 (None, '')。"""
    if not path:
        return None, ""
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_EXT:
        raise gr.Error(f"不支援的格式 {ext}，請上傳 PNG / JPG / JPEG / BMP。")
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise gr.Error("檔案無法解碼，可能已損毀。")
    if raw.dtype != np.uint8:  # 16-bit 等轉成 8-bit
        raw = cv2.convertScaleAbs(raw, alpha=255.0 / max(raw.max(), 1))
    channels = 1 if raw.ndim == 2 else raw.shape[2]
    h, w = raw.shape[:2]
    if channels == 1:
        rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    elif channels == 4:
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    note = ""
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        note = f"\n- ⚠️ 影像過大，已縮小為 {rgb.shape[1]}×{rgb.shape[0]} 進行處理"
    size_kb = os.path.getsize(path) / 1024
    info = (
        f"- **格式**：{ext[1:].upper()}\n"
        f"- **解析度**：{w} × {h} 像素\n"
        f"- **色彩維度**：{channels} 通道（{ {1: '灰階', 3: 'BGR 彩色', 4: 'BGRA 含透明度'}.get(channels, '?') }），每通道 8 bit\n"
        f"- **檔案大小**：{size_kb:.1f} KB"
        f"{note}"
    )
    return rgb, info


def _need_image(img):
    if img is None:
        raise gr.Error("請先在上方上傳一張影像。")


# ---------------------------------------------------------------- 模組 2
def rgb_to_yuv(rgb):
    """BT.601 full-range (JPEG) YCbCr，U/V 已加 128 偏移，值域 0–255。"""
    f = rgb.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    v = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
    return [np.clip(c, 0, 255).round().astype(np.uint8) for c in (y, u, v)]


def _hist(ch):
    return np.bincount(ch.ravel(), minlength=256)


def color_view(img, mode):
    _need_image(img)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(256)
    if mode == "Original":
        shown = img
        for i, (c, name) in enumerate(zip(("#d62728", "#2ca02c", "#1f77b4"), "RGB")):
            ax.fill_between(x, _hist(img[..., i]), alpha=0.45, color=c, label=name)
        ax.legend()
    else:
        yuv = rgb_to_yuv(img)
        chans = {"R": img[..., 0], "G": img[..., 1], "B": img[..., 2],
                 "Y": yuv[0], "U": yuv[1], "V": yuv[2]}
        ch = chans[mode]
        shown = ch
        ax.bar(x, _hist(ch), width=1.0, color="#555555")
    ax.set_xlim(0, 255)
    ax.set_xlabel("Pixel intensity (0–255)")
    ax.set_ylabel("Pixel count")
    ax.set_title(f"Histogram – {mode}")
    fig.tight_layout()
    return shown, fig


# ---------------------------------------------------------------- 模組 3
PRESETS = {
    "自訂": None,
    "恆等 (Identity)": "0 0 0\n0 1 0\n0 0 0",
    "平均模糊 3×3": "1 1 1\n1 1 1\n1 1 1",
    "高斯平滑 3×3": "1 2 1\n2 4 2\n1 2 1",
    "高斯平滑 5×5": "1 4 6 4 1\n4 16 24 16 4\n6 24 36 24 6\n4 16 24 16 4\n1 4 6 4 1",
    "銳化 (Sharpen)": "0 -1 0\n-1 5 -1\n0 -1 0",
    "邊緣偵測 (Laplacian)": "-1 -1 -1\n-1 8 -1\n-1 -1 -1",
    "浮雕 (Emboss)": "-2 -1 0\n-1 1 1\n0 1 2",
    "Sobel X": "-1 0 1\n-2 0 2\n-1 0 1",
}


def apply_preset(name):
    return PRESETS[name] if PRESETS.get(name) else gr.update()


def parse_kernel(text):
    """文字 -> 2D float 陣列，含完整防呆。"""
    rows = [r for r in (text or "").replace(";", "\n").strip().splitlines() if r.strip()]
    if not rows:
        raise gr.Error("卷積核是空的，請輸入矩陣（每列一行，數值以空白或逗號分隔）。")
    try:
        mat = [[float(t) for t in r.replace(",", " ").split()] for r in rows]
    except ValueError:
        raise gr.Error("卷積核含有非數值內容，請只輸入數字。")
    if len({len(r) for r in mat}) != 1:
        raise gr.Error("各列的元素數量不一致，請輸入完整矩陣。")
    k = np.array(mat, dtype=np.float64)
    h, w = k.shape
    if h != w:
        raise gr.Error(f"目前為 {h}×{w} 非方陣，請輸入方陣（如 3×3、5×5）。")
    if h % 2 == 0:
        raise gr.Error(f"{h}×{h} 為偶數尺寸，卷積核需為奇數尺寸才有中心點。")
    if h > MAX_KERNEL:
        raise gr.Error(f"卷積核最大支援 {MAX_KERNEL}×{MAX_KERNEL}。")
    if not np.isfinite(k).all():
        raise gr.Error("卷積核含有 NaN 或無限大，請修正。")
    return k


def convolve2d(img, k):
    """自行實作的 2D 卷積（核翻轉 180°，邊界以 edge 複製填補）。"""
    p = k.shape[0] // 2
    f = img.astype(np.float32)
    padded = np.pad(f, ((p, p), (p, p), (0, 0)), mode="edge")
    kf = k[::-1, ::-1]
    h, w = f.shape[:2]
    out = np.zeros_like(f)
    for i in range(k.shape[0]):
        for j in range(k.shape[1]):
            if kf[i, j] != 0:
                out += np.float32(kf[i, j]) * padded[i:i + h, j:j + w]
    return out


def psnr(a, b, max_val=255.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return mse, float("inf")
    return mse, 10 * np.log10(max_val ** 2 / mse)


def run_filter(img, text, normalize, offset):
    _need_image(img)
    k = parse_kernel(text)
    notes = []
    if normalize:
        s = k.sum()
        if abs(s) < 1e-12:
            notes.append("⚠️ 核的總和為 0（如邊緣/浮雕核），無法正規化，已略過。")
        else:
            k = k / s
            notes.append(f"已正規化（除以總和 {s:g}）。")
    out = convolve2d(img, k) + float(offset or 0)
    out = np.clip(out, 0, 255).round().astype(np.uint8)

    # 與 OpenCV 比對，驗證自行實作的正確性
    ref = cv2.filter2D(img.astype(np.float32), -1, np.ascontiguousarray(k[::-1, ::-1]).astype(np.float32),
                       borderType=cv2.BORDER_REPLICATE) + float(offset or 0)
    ref = np.clip(ref, 0, 255).round().astype(np.uint8)
    max_diff = int(np.abs(out.astype(int) - ref.astype(int)).max())

    mse, val = psnr(img, out)
    psnr_txt = "∞ dB（兩張影像完全相同，MSE = 0）" if np.isinf(val) else f"{val:.2f} dB"
    report = (
        f"### PSNR：{psnr_txt}\n"
        f"- MSE：{mse:.4f}\n"
        f"- 公式：PSNR = 10·log₁₀(255² / MSE)，MSE = 1/(H·W·C)·Σ(原圖 − 濾波後)²\n"
        f"- 與 `cv2.filter2D` 最大像素差：{max_diff}\n"
        + "".join(f"- {n}\n" for n in notes)
    )
    return out, report


# ---------------------------------------------------------------- 模組 4
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # Colab / Jupyter 儲存格中沒有 __file__
    _HERE = os.getcwd()
YUNET_FILE = os.path.join(_HERE, "face_detection_yunet_2023mar.onnx")
YUNET_URLS = [
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
]
_HAAR = None


def _ensure_yunet():
    """確保 YuNet 模型存在（不存在時嘗試自動下載），回傳路徑或 None。"""
    if os.path.exists(YUNET_FILE) and os.path.getsize(YUNET_FILE) > 100_000:
        return YUNET_FILE
    import urllib.request
    for url in YUNET_URLS:
        try:
            urllib.request.urlretrieve(url, YUNET_FILE)
            if os.path.getsize(YUNET_FILE) > 100_000:
                return YUNET_FILE
            os.remove(YUNET_FILE)  # 下載到的是 LFS 指標檔等無效內容
        except Exception:
            continue
    return None


def _detect_yunet(img, min_conf):
    path = _ensure_yunet()
    if path is None or not hasattr(cv2, "FaceDetectorYN"):
        return None
    h, w = img.shape[:2]
    det = cv2.FaceDetectorYN.create(path, "", (w, h), float(min_conf), 0.3, 5000)
    _, faces = det.detect(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    rows = []
    for f in (faces if faces is not None else []):
        x, y, fw, fh = f[:4]
        x1, y1 = max(int(x), 0), max(int(y), 0)
        x2, y2 = min(int(x + fw), w - 1), min(int(y + fh), h - 1)
        rows.append((x1, y1, x2, y2, float(f[-1])))
    return rows, "YuNet（輕量級 CNN 人臉偵測器）"


def _detect_haar(img, min_conf):
    global _HAAR
    if _HAAR is None:
        data_dir = getattr(getattr(cv2, "data", None), "haarcascades", "")
        _HAAR = cv2.CascadeClassifier(os.path.join(data_dir, "haarcascade_frontalface_default.xml"))
    if _HAAR.empty():
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    boxes, _, weights = _HAAR.detectMultiScale3(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40), outputRejectLevels=True)
    rows = []
    for (x, y, fw, fh), wgt in zip(boxes, np.ravel(weights)):
        conf = float(1 / (1 + np.exp(-float(wgt))))  # levelWeights 經 sigmoid，近似信心度
        if conf >= float(min_conf):
            rows.append((int(x), int(y), int(x + fw), int(y + fh), conf))
    return rows, "Haar-Cascade（信心度為近似值）"


def detect_faces(img, min_conf):
    _need_image(img)
    result = _detect_yunet(img, min_conf) or _detect_haar(img, min_conf)
    if result is None:
        raise gr.Error(
            "找不到可用的人臉偵測模型。請安裝 opencv-python-headless<5，"
            "或手動下載 face_detection_yunet_2023mar.onnx 放到 app.py 同一資料夾。")
    rows, engine = result
    canvas = img.copy()
    h, w = img.shape[:2]
    for i, (x1, y1, x2, y2, c) in enumerate(rows, 1):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), max(2, w // 400))
        cv2.putText(canvas, f"#{i} {c:.2f}", (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.5, w / 1200), (0, 255, 0), 2)
    table = [[i, x1, y1, x2 - x1, y2 - y1, round(c, 3)] for i, (x1, y1, x2, y2, c) in enumerate(rows, 1)]
    msg = f"### 偵測到 {len(rows)} 張人臉\n引擎：{engine}"
    if not rows:
        msg += "\n\n未偵測到人臉，可嘗試調低信心度門檻或更換影像。"
    return canvas, msg, table


# ---------------------------------------------------------------- UI
with gr.Blocks(title="影像處理與人臉分析工具") as demo:
    gr.Markdown("# 影像處理與人臉分析工具\n上傳一張影像，再切換到各分頁操作。")
    state = gr.State(None)
    upload = gr.Image(type="filepath", sources=["upload"], label="上傳影像（PNG / JPG / BMP）", height=300)
    info = gr.Markdown()
    upload.change(load_image, upload, [state, info])

    with gr.Tab("色彩空間與直方圖"):
        mode = gr.Radio(["Original", "R", "G", "B", "Y", "U", "V"], value="Original", label="顯示模式")
        with gr.Row():
            view = gr.Image(label="影像", interactive=False)
            hist = gr.Plot(label="直方圖")
        # 切換模式、或重新載入影像時同步更新
        mode.change(color_view, [state, mode], [view, hist])
        state.change(lambda i, m: color_view(i, m) if i is not None else (None, None),
                     [state, mode], [view, hist])

    with gr.Tab("自訂卷積濾波與 PSNR"):
        preset = gr.Dropdown(list(PRESETS), value="高斯平滑 3×3", label="預設核（可選）")
        kernel = gr.Textbox(PRESETS["高斯平滑 3×3"], lines=6, label="卷積核（每列一行，數值以空白或逗號分隔）")
        with gr.Row():
            norm = gr.Checkbox(True, label="正規化（除以核總和）")
            offs = gr.Number(0, label="輸出偏移量（浮雕可設 128）")
        run = gr.Button("套用濾波", variant="primary")
        with gr.Row():
            f_in = gr.Image(label="濾波前", interactive=False)
            f_out = gr.Image(label="濾波後", interactive=False)
        report = gr.Markdown()
        preset.change(apply_preset, preset, kernel)
        run.click(lambda i, *a: (i, *run_filter(i, *a)), [state, kernel, norm, offs], [f_in, f_out, report])

    with gr.Tab("人臉偵測"):
        conf = gr.Slider(0.1, 0.95, 0.5, step=0.05, label="最低信心度門檻")
        det_btn = gr.Button("開始偵測", variant="primary")
        det_img = gr.Image(label="偵測結果", interactive=False)
        det_msg = gr.Markdown()
        det_tbl = gr.Dataframe(headers=["#", "x", "y", "寬", "高", "信心度"], interactive=False)
        det_btn.click(detect_faces, [state, conf], [det_img, det_msg, det_tbl])

if __name__ == "__main__":
    import sys
    # Colab 需要 share=True 才能取得公開連結；本機與 Hugging Face 則不需要
    demo.launch(share="google.colab" in sys.modules)
