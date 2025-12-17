import sys
import asyncio
import os
import tempfile
import shutil
import cv2
import numpy as np
import matplotlib

# Sử dụng backend không tương tác
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import SSIM
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

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

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def silence_event_loop_closed(loop, context):
    exc = context.get('exception')
    if isinstance(exc, ConnectionResetError) or (isinstance(exc, OSError) and exc.winerror == 10054):
        return 
try:
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(silence_event_loop_closed)
except RuntimeError:
    pass

try:
    from asyncio.proactor_events import ProactorBasePipeTransport
    _orig_call = ProactorBasePipeTransport._call_connection_lost
    def _safe_call(self, *args, **kwargs):
        try:
            return _orig_call(self, *args, **kwargs)
        except (ConnectionResetError, OSError):
            return None
    ProactorBasePipeTransport._call_connection_lost = _safe_call
except Exception:
    pass

try:
    import gradio as gr
except Exception:
    gr = None


class FeatureMatcher:
    def __init__(self, method="ORB"):
        self.method = method
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        if method == "SIFT":
            self.detector = cv2.SIFT_create(nfeatures=5000)
            self.norm = cv2.NORM_L2
        else: # ORB
            self.detector = cv2.ORB_create(
                nfeatures=5000,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=15,
                firstLevel=0,
                WTA_K=2,
                scoreType=cv2.ORB_HARRIS_SCORE,
                patchSize=31,
                fastThreshold=5 
            )
            self.norm = cv2.NORM_HAMMING

        self.matcher = cv2.BFMatcher(self.norm, crossCheck=False)

    def match(self, img1_gray, img2_gray, max_matches=500):
        img1_enhanced = self.clahe.apply(img1_gray)
        img2_enhanced = self.clahe.apply(img2_gray)

        kp1, des1 = self.detector.detectAndCompute(img1_enhanced, None)
        kp2, des2 = self.detector.detectAndCompute(img2_enhanced, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return [], [], []

        matches = self.matcher.knnMatch(des1, des2, k=2)
        good = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.8 * n.distance:
                    good.append(m)

        good = sorted(good, key=lambda x: x.distance)[:max_matches]
        pts1 = [kp1[m.queryIdx].pt for m in good]
        pts2 = [kp2[m.trainIdx].pt for m in good]
        return pts1, pts2, good


class MotionEstimator:
    def __init__(self, ransac_thresh=10.0, max_rotation=0.1): 
        self.ransac_thresh = ransac_thresh
        self.max_rotation = max_rotation 

    def estimate(self, pts1, pts2):
        if len(pts1) < 4:
            return 0.0, 0.0, 0.0

        pts1 = np.float32(pts1).reshape(-1, 1, 2)
        pts2 = np.float32(pts2).reshape(-1, 1, 2)

        M, inliers = cv2.estimateAffinePartial2D(
            pts1, pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
            confidence=0.99,
            maxIters=2000
        )

        if M is None or inliers is None or inliers.sum() < 5:
            return 0.0, 0.0, 0.0

        dx = M[0, 2]
        dy = M[1, 2]
        da = np.arctan2(M[1, 0], M[0, 0])
        da = np.clip(da, -self.max_rotation, self.max_rotation)
        return dx, dy, da


class TrajectorySmoother:
    def __init__(self, radius_trans=30, radius_rot=10):
        self.radius_trans = int(radius_trans)
        self.radius_rot = int(radius_rot)

    def _smooth_1d(self, curve, radius):
        window_size = 2 * radius + 1
        window = np.ones(window_size)
        window /= window.sum()
        curve_pad = np.pad(curve, (radius, radius), mode='edge')
        return np.convolve(curve_pad, window, mode='same')[radius:-radius]

    def smooth(self, trajectory):
        smoothed = np.copy(trajectory)
        smoothed[:, 0] = self._smooth_1d(trajectory[:, 0], self.radius_trans)
        smoothed[:, 1] = self._smooth_1d(trajectory[:, 1], self.radius_trans)
        smoothed[:, 2] = self._smooth_1d(trajectory[:, 2], self.radius_rot)
        return smoothed

# ==========================================
# 3. CÁC HÀM HỖ TRỢ
# ==========================================
def save_output_locally(temp_path):
    try:
        current_dir = os.getcwd()
        import re
        existing = [f for f in os.listdir(current_dir) if f.startswith("output_") and f.endswith(".mp4")]
        nums = []
        for f in existing:
            m = re.match(r"output_(\d+)\.mp4$", f)
            if m:
                try: nums.append(int(m.group(1)))
                except: pass
        next_num = max(nums) + 1 if nums else 1
        filename = f"output_{next_num:03d}.mp4"
        local_path = os.path.join(current_dir, filename)
        shutil.copy(temp_path, local_path)
        return filename
    except Exception:
        return None

def compute_metrics_interframe(img1, img2):
    try:
        h, w = 90, 160 # Resize nhỏ để tính nhanh
        i1 = cv2.resize(img1, (w, h))
        i2 = cv2.resize(img2, (w, h))
        g1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
        
        p = cv2.PSNR(g1, g2)
        s = 0
        if HAS_SKIMAGE:
            try: s = ssim(g1, g2, data_range=255)
            except: s = 0
        return p, s
    except:
        return 0, 0

# ==========================================
# 4. ĐỘNG CƠ XỬ LÝ CHÍNH
# ==========================================
def run_stabilizer_engine(input_path, tech_method, smoothing_radius, zoom_ratio, border_strategy, is_preview=False, progress=gr.Progress()):
    if input_path is None:
        # Trả về mặc định 0 nếu không có video
        return None, None, None, "⚠️ Vui lòng tải video lên!", "0", "0", "0", "0"

    try:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(silence_event_loop_closed)
    except RuntimeError:
        pass

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): return None, None, None, "❌ Lỗi đọc file!", "0", "0", "0", "0"
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if is_preview:
        n_frames = min(total_frames, 150)
        task_name = "PREVIEW"
    else:
        n_frames = total_frames
        task_name = "FULL RENDER"

    # --- SETUP ---
    matcher = FeatureMatcher(method=("SIFT" if "SIFT" in tech_method else "ORB"))
    estimator = MotionEstimator()
    smoother = TrajectorySmoother(radius_trans=smoothing_radius, radius_rot=max(1, smoothing_radius // 3))

    fd, out_path = tempfile.mkstemp(suffix='.mp4')
    os.close(fd)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # ================= PHASE 1 =================
    transforms = []
    _, prev = cap.read()
    prev_small = cv2.resize(prev, (640, int(360 * h / w)))
    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
    scale_x = w / prev_small.shape[1]
    scale_y = h / prev_small.shape[0]

    limit_analyze = n_frames - 1
    for i in range(limit_analyze):
        progress((i + 1) / limit_analyze, desc=f"[{task_name}] Phase 1: Phân tích")
        success, curr = cap.read()
        if not success: break
        
        curr_small = cv2.resize(curr, (prev_small.shape[1], prev_small.shape[0]))
        curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
        
        pts1, pts2, _ = matcher.match(prev_gray, curr_gray)
        dx, dy, da = estimator.estimate(pts1, pts2)
        transforms.append([dx * scale_x, dy * scale_y, da])
        prev_gray = curr_gray

    if len(transforms) == 0:
        return None, None, None, "⚠️ Không phát hiện đủ chuyển động!", "0", "0", "0", "0"

    transforms = np.array(transforms, dtype=np.float32)
    trajectory = np.cumsum(transforms, axis=0)
    trajectory = np.insert(trajectory, 0, [0,0,0], axis=0)
    
    smoothed_trajectory = smoother.smooth(trajectory)
    difference = smoothed_trajectory - trajectory

    # ================= PHASE 2 =================
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cx, cy = w / 2, h / 2
    final_scale = (1.0 + zoom_ratio) if "Zoom" in border_strategy else 1.0
    total_render = min(len(difference), n_frames)
    
    val_psnr_in, val_ssim_in = [], []
    val_psnr_out, val_ssim_out = [], []
    
    _, prev_frame_in = cap.read()
    # Frame 0 setup
    dx0, dy0, da0 = difference[0]
    M0 = np.array([
        [final_scale*np.cos(da0), -final_scale*np.sin(da0), (1-final_scale*np.cos(da0))*cx + final_scale*np.sin(da0)*cy + dx0],
        [final_scale*np.sin(da0),  final_scale*np.cos(da0), -final_scale*np.sin(da0)*cx + (1-final_scale*np.cos(da0))*cy + dy0]
    ], dtype=np.float32)
    prev_frame_out = cv2.warpAffine(prev_frame_in, M0, (w, h), flags=cv2.INTER_LINEAR)
    out.write(prev_frame_out)

    for i in range(1, total_render):
        progress((i + 1) / total_render, desc=f"[{task_name}] Phase 2: Xuất video")
        success, curr_frame_in = cap.read()
        if not success: break

        dx, dy, da = difference[int(i)]
        alpha = final_scale * np.cos(da)
        beta = final_scale * np.sin(da)
        M_final = np.array([
            [alpha, -beta, (1-alpha)*cx + beta*cy + dx],
            [beta,  alpha, -beta*cx + (1-alpha)*cy + dy]
        ], dtype=np.float32)

        border_mode = cv2.BORDER_REPLICATE if "Replicate" in border_strategy else cv2.BORDER_CONSTANT
        curr_frame_out = cv2.warpAffine(curr_frame_in, M_final, (w, h), flags=cv2.INTER_LINEAR, borderMode=border_mode)
        out.write(curr_frame_out)
        
        # Chỉ số Metrics (Tính mỗi 2 frame)
        if i % 2 == 0:
            pi, si = compute_metrics_interframe(prev_frame_in, curr_frame_in)
            val_psnr_in.append(pi); val_ssim_in.append(si)
            
            po, so = compute_metrics_interframe(prev_frame_out, curr_frame_out)
            val_psnr_out.append(po); val_ssim_out.append(so)
        else:
            if val_psnr_in:
                val_psnr_in.append(val_psnr_in[-1]); val_ssim_in.append(val_ssim_in[-1])
                val_psnr_out.append(val_psnr_out[-1]); val_ssim_out.append(val_ssim_out[-1])

        prev_frame_in = curr_frame_in
        prev_frame_out = curr_frame_out

    cap.release()
    out.release()
    
    # --- VẼ ĐỒ THỊ ---
    fig_traj = plt.figure(figsize=(10, 4))
    plt.plot(trajectory[:, 0], label='Input', alpha=0.5, color='gray')
    plt.plot(smoothed_trajectory[:, 0], label='Output', color='red', linewidth=2)
    plt.title("Quỹ đạo"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    
    fig_metric, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(val_psnr_in, label='In', color='gray', linestyle='--'); ax1.plot(val_psnr_out, label='Out', color='green')
    ax1.set_ylabel('PSNR'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(val_ssim_in, label='In', color='gray', linestyle='--'); ax2.plot(val_ssim_out, label='Out', color='blue')
    ax2.set_ylabel('SSIM'); ax2.legend(); ax2.grid(True, alpha=0.3); plt.tight_layout()

    # --- TÍNH TRUNG BÌNH (ĐÂY LÀ PHẦN BẠN CẦN) ---
    avg_psnr_in = np.mean(val_psnr_in) if val_psnr_in else 0
    avg_ssim_in = np.mean(val_ssim_in) if val_ssim_in else 0
    avg_psnr_out = np.mean(val_psnr_out) if val_psnr_out else 0
    avg_ssim_out = np.mean(val_ssim_out) if val_ssim_out else 0

    status_msg = f"✅ Hoàn tất {task_name}!"
    if not is_preview:
        saved_name = save_output_locally(out_path)
        if saved_name: status_msg += f"\n📂 Đã lưu: {saved_name}"

    # Trả về thêm 4 giá trị trung bình
    return (
        out_path, 
        fig_traj, 
        fig_metric, 
        status_msg, 
        f"{avg_psnr_in:.2f}", 
        f"{avg_ssim_in:.4f}", 
        f"{avg_psnr_out:.2f}", 
        f"{avg_ssim_out:.4f}"
    )

# ==========================================
# 5. GIAO DIỆN GRADIO
# ==========================================
def run_preview(vid, method, rad, border_strat, zoom):
    z_val = zoom if "Zoom" in border_strat else 0.0
    return run_stabilizer_engine(vid, method, rad, z_val, border_strat, is_preview=True)

def run_full(vid, method, rad, border_strat, zoom):
    z_val = zoom if "Zoom" in border_strat else 0.0
    return run_stabilizer_engine(vid, method, rad, z_val, border_strat, is_preview=False)

def toggle_zoom_slider(strategy):
    if "Zoom" in strategy: return gr.update(visible=True)
    else: return gr.update(visible=False)

with gr.Blocks(title="Video Stabilization Pro", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎥 Đồ án: Ổn định Video Kỹ thuật số")
    
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Video(label="Input Video") 
            with gr.Group():
                tech = gr.Dropdown(["Optical Flow (Classic)", "ORB (Feature Matching)", "SIFT (High Accuracy)"], value="ORB (Feature Matching)", label="1. Kỹ thuật")
                rad = gr.Slider(5, 100, value=30, label="2. Độ mượt")
                border_strat = gr.Radio(["Zoom & Crop", "Replicate Border"], value="Zoom & Crop", label="3. Xử lý viền")
                zoom = gr.Slider(0.0, 0.3, value=0.1, label="Tỷ lệ Zoom", visible=True)
            
            with gr.Row():
                btn_prev = gr.Button("👁️ Xem thử", variant="secondary")
                btn_full = gr.Button("🚀 Xử lý Full", variant="primary")
            status_output = gr.Textbox(label="Trạng thái", interactive=False)

        with gr.Column(scale=1):
            out_vid = gr.Video(label="Kết quả Output")
            
            # --- PHẦN HIỂN THỊ CHỈ SỐ CŨ (ĐÃ KHÔI PHỤC) ---
            gr.Markdown("### 📊 Chỉ số Trung bình (Average Metrics)")
            with gr.Row():
                with gr.Group():
                    gr.Markdown("**Gốc (Input)**")
                    t_pi = gr.Textbox(label="PSNR", value="0")
                    t_si = gr.Textbox(label="SSIM", value="0")
                with gr.Group():
                    gr.Markdown("**Đã xử lý (Output)**")
                    t_po = gr.Textbox(label="PSNR", value="0")
                    t_so = gr.Textbox(label="SSIM", value="0")
            
            with gr.Tab("Quỹ đạo"): plot_traj = gr.Plot()
            with gr.Tab("Đồ thị Chi tiết"): plot_metric = gr.Plot()
    
    border_strat.change(fn=toggle_zoom_slider, inputs=border_strat, outputs=zoom)
    
    # Cập nhật outputs để bao gồm 4 ô Textbox mới
    outputs_list = [out_vid, plot_traj, plot_metric, status_output, t_pi, t_si, t_po, t_so]
    
    btn_prev.click(run_preview, inputs=[inp, tech, rad, border_strat, zoom], outputs=outputs_list)
    btn_full.click(run_full, inputs=[inp, tech, rad, border_strat, zoom], outputs=outputs_list)

if __name__ == "__main__":
    demo.launch(share=False)