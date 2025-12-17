import sys
import asyncio
import os
import tempfile
import shutil
from datetime import datetime
import cv2
import numpy as np
import matplotlib
# Use non-interactive backend to avoid GUI issues in worker threads
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Optional: filter noisy WinError 10054 tracebacks from stderr to reduce log spam
try:
    _orig_stderr_write = sys.stderr.write
    def _stderr_filter(s):
        try:
            if isinstance(s, str) and ("WinError 10054" in s or "ProactorBasePipeTransport._call_connection_lost" in s or "_call_connection_lost" in s):
                return 0
        except Exception:
            pass
        return _orig_stderr_write(s)
    sys.stderr.write = _stderr_filter
except Exception:
    pass

# ==========================================
# 1. KHẮC PHỤC TRIỆT ĐỂ LỖI WINERROR 10054
# ==========================================
# Bước 1: Thay đổi chính sách Event Loop ngay khi khởi động
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Bước 2: Định nghĩa hàm lọc lỗi
def silence_event_loop_closed(loop, context):
    exc = context.get('exception')
    # Nếu là lỗi ngắt kết nối (10054) -> Bỏ qua, không in ra màn hình
    if isinstance(exc, ConnectionResetError) or (isinstance(exc, OSError) and exc.winerror == 10054):
        return 
    # Các lỗi khác vẫn in ra để debug
    # loop.default_exception_handler(context)

# Thiết lập handler cho loop hiện tại (nếu có) để chặn lỗi WinError 10054 toàn cục
try:
    # Prefer get_running_loop to avoid DeprecationWarning when no loop exists
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(silence_event_loop_closed)
except RuntimeError:
    # No running loop in this context; skip setting handler now
    pass

# Monkey-patch ProactorBasePipeTransport._call_connection_lost để tránh traceback noisy trên Windows
try:
    from asyncio.proactor_events import ProactorBasePipeTransport
    _orig_call = ProactorBasePipeTransport._call_connection_lost
    def _safe_call(self, *args, **kwargs):
        try:
            return _orig_call(self, *args, **kwargs)
        except ConnectionResetError:
            return None
        except OSError as e:
            if getattr(e, 'winerror', None) == 10054:
                return None
            raise
    ProactorBasePipeTransport._call_connection_lost = _safe_call
except Exception:
    pass

# Import Gradio after setting event loop policy and monkeypatch to avoid Proactor creation
try:
    import gradio as gr
except Exception:
    gr = None

# ==========================================
# 2. CÁC HÀM HỖ TRỢ (UTILS)
# ==========================================
def save_output_locally(temp_path):
    """Lưu file từ Temp ra thư mục hiện tại."""
    try:
        current_dir = os.getcwd()
        # Tạo tên file tăng dần: output_001.mp4, output_002.mp4, ...
        import re
        existing = [f for f in os.listdir(current_dir) if f.startswith("output_") and f.endswith(".mp4")]
        nums = []
        for f in existing:
            m = re.match(r"output_(\d+)\.mp4$", f)
            if m:
                try:
                    nums.append(int(m.group(1)))
                except Exception:
                    pass
        next_num = max(nums) + 1 if nums else 1
        filename = f"output_{next_num:03d}.mp4"
        local_path = os.path.join(current_dir, filename)
        shutil.copy(temp_path, local_path)
        return filename
    except Exception:
        return None

def moving_average(curve, radius):
    """Làm mượt quỹ đạo (Smoothing)."""
    radius = int(radius)
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    return curve_smoothed[radius:-radius]

def compute_psnr_quick(img1, img2):
    """Tính PSNR nhanh trên ảnh nhỏ."""
    s1 = cv2.resize(img1, (160, 90))
    s2 = cv2.resize(img2, (160, 90))
    return cv2.PSNR(s1, s2)

# ==========================================
# 3. ĐỘNG CƠ XỬ LÝ CHÍNH (CORE ENGINE)
# ==========================================
def run_stabilizer_engine(input_path, tech_method, smoothing_radius, zoom_factor, is_preview=False, progress=gr.Progress()):
    if input_path is None:
        return None, None, None, "⚠️ Vui lòng tải video lên!"

    # --- [QUAN TRỌNG] ÁP DỤNG BỘ LỌC LỖI VÀO LOOP HIỆN TẠI ---
    try:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(silence_event_loop_closed)
    except RuntimeError:
        pass

    # --- ĐỌC VIDEO ---
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): return None, None, None, "❌ Lỗi đọc file!"
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # --- CẤU HÌNH CHẾ ĐỘ ---
    if is_preview:
        n_frames = min(total_frames, 150) # Preview 5s
        task_name = "PREVIEW (Xem thử)"
    else:
        n_frames = total_frames
        task_name = "FULL RENDER (Xử lý)"

    # --- TẠO FILE TẠM ---
    fd, out_path = tempfile.mkstemp(suffix='.mp4')
    os.close(fd) # Đóng file ngay để Windows không khóa
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # --- CẤU HÌNH THUẬT TOÁN ---
    process_w = 640
    scale = w / process_w
    process_h = int(h / scale)

    detector = None
    matcher = None
    if "ORB" in tech_method:
        detector = cv2.ORB_create(nfeatures=1000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif "SIFT" in tech_method:
        detector = cv2.SIFT_create(nfeatures=1000)
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    # ================= PHASE 1: PHÂN TÍCH CHUYỂN ĐỘNG =================
    # Use a dynamic list to avoid index errors when frame counts vary
    transforms = []
    _, prev = cap.read()
    prev_small = cv2.resize(prev, (process_w, process_h))
    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)

    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

    for i in progress.tqdm(range(n_frames - 2), desc=f"[{task_name}] Phase 1: Analyzing"):
        success, curr = cap.read()
        if not success: break
        
        curr_small = cv2.resize(curr, (process_w, process_h))
        curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
        
        # Logic Feature Matching / Optical Flow
        src_pts, dst_pts = None, None
        if "Optical Flow" in tech_method:
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
            if p0 is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
                src_pts = p0[st==1]; dst_pts = p1[st==1]
        else:
            kp1, des1 = detector.detectAndCompute(prev_gray, None)
            kp2, des2 = detector.detectAndCompute(curr_gray, None)
            if des1 is not None and des2 is not None and len(des1)>0 and len(des2)>0:
                matches = matcher.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)[:100]
                if len(matches) > 10:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

        if src_pts is not None and dst_pts is not None and len(src_pts) > 10:
            m, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            if m is not None:
                transforms.append([m[0, 2] * scale, m[1, 2] * scale, np.arctan2(m[1, 0], m[0, 0])])
            else:
                transforms.append([0.0, 0.0, 0.0])
        else:
            transforms.append([0.0, 0.0, 0.0])
        
        prev_gray = curr_gray

    # ================= PHASE 2: LÀM MƯỢT & XUẤT VIDEO =================
    # Convert transforms list to numpy array (N x 3)
    transforms = np.array(transforms, dtype=np.float32)
    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = np.copy(trajectory)
    for j in range(3):
        smoothed_trajectory[:, j] = moving_average(trajectory[:, j], radius=smoothing_radius)
    difference = smoothed_trajectory - trajectory

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    psnr_vals = []
    
    T_zoom = cv2.getRotationMatrix2D((w/2, h/2), 0, zoom_factor)
    M_zoom = np.vstack([T_zoom, [0, 0, 1]])

    # Use actual number of computed transforms to drive rendering
    for i in progress.tqdm(range(len(transforms)), desc=f"[{task_name}] Phase 2: Rendering"):
        success, frame = cap.read()
        if not success: break

        dx, dy, da = difference[i]
        alpha = -da * 180 / np.pi
        
        T_stab = cv2.getRotationMatrix2D((w/2, h/2), alpha, 1.0)
        T_stab[0, 2] += dx; T_stab[1, 2] += dy
        
        M_final = np.dot(M_zoom, np.vstack([T_stab, [0, 0, 1]]))
        frame_final = cv2.warpAffine(frame, M_final[:2, :], (w, h), borderMode=cv2.BORDER_REFLECT)
        out.write(frame_final)
        
        if i % 2 == 0: psnr_vals.append(compute_psnr_quick(frame, frame_final))
        else: psnr_vals.append(psnr_vals[-1] if psnr_vals else 0)

    cap.release()
    out.release()
    
    # ================= KẾT THÚC =================
    fig_traj = plt.figure(figsize=(10, 4))
    plt.plot(trajectory[:, 0], label='Original', alpha=0.5)
    plt.plot(smoothed_trajectory[:, 0], label='Smoothed', color='red', linewidth=2)
    plt.title("Quỹ đạo Chuyển động (Trajectory)"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fig_metric = plt.figure(figsize=(10, 4))
    plt.plot(psnr_vals, color='green'); plt.title("Chất lượng PSNR"); plt.grid(True, alpha=0.3); plt.tight_layout()

    status_msg = f"✅ Hoàn tất {task_name}!"
    if not is_preview:
        saved_name = save_output_locally(out_path)
        if saved_name: status_msg += f"\n📂 Đã lưu file: {saved_name}"
    else:
        status_msg += "\n(Chế độ xem trước: Không lưu file)"

    return out_path, fig_traj, fig_metric, status_msg

# ==========================================
# 4. WRAPPERS & GRADIO UI
# ==========================================
def run_preview(vid, method, rad, zoom):
    return run_stabilizer_engine(vid, method, rad, zoom, is_preview=True)

def run_full(vid, method, rad, zoom):
    return run_stabilizer_engine(vid, method, rad, zoom, is_preview=False)

with gr.Blocks(title="Video Stabilization Project") as demo:
    gr.Markdown("# 🎥 Đồ án: Ổn định Video Kỹ thuật số")
    
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Video(label="Input Video", sources=["upload"])
            with gr.Group():
                tech = gr.Dropdown(
                    choices=["Optical Flow (Lucas-Kanade)", "ORB", "SIFT"],
                    value="Optical Flow (Lucas-Kanade)", label="Kỹ thuật Feature Matching"
                )
                rad = gr.Slider(10, 100, value=30, step=1, label="Smoothing Radius")
                zoom = gr.Slider(1.0, 1.3, value=1.05, step=0.01, label="Zoom (Crop viền)")
            
            with gr.Row():
                btn_prev = gr.Button("👁️ Xem thử (5 giây)", variant="secondary")
                btn_full = gr.Button("🚀 Xử lý & Tải về", variant="primary")
            
            status_output = gr.Textbox(label="Trạng thái", interactive=False)

        with gr.Column(scale=1):
            out_vid = gr.Video(label="Kết quả Output")
            with gr.Tab("Quỹ đạo"): plot_traj = gr.Plot()
            with gr.Tab("Chỉ số PSNR"): plot_metric = gr.Plot()
    
    btn_prev.click(run_preview, inputs=[inp, tech, rad, zoom], outputs=[out_vid, plot_traj, plot_metric, status_output])
    btn_full.click(run_full, inputs=[inp, tech, rad, zoom], outputs=[out_vid, plot_traj, plot_metric, status_output])

if __name__ == "__main__":
    demo.launch(share=False)