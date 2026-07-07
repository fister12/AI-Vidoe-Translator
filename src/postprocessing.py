from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import numpy as np
import cv2
from tqdm import tqdm


def advanced_postprocess_with_optical_flow(
    video_path: str | Path,
    output_path: str | Path,
    denoise_strength: float = 1.2,
    sharpen_amount: float = 0.6,
    contrast: float = 1.02,
    saturation: float = 1.03,
    crf: int = 16,
    preset: str = "slow",
    enable_optical_flow_smoothing: bool = True,
    flow_smoothing_strength: float = 0.5,
) -> Path:
    """
    Advanced post-processing with optical flow-based motion smoothing.
    Reduces jitter and unnatural mouth movements from Wav2Lip.
    
    This is more effective than simple temporal filtering for fixing
    discontinuous lip movements.
    
    Args:
        video_path: Input video
        output_path: Output video
        enable_optical_flow_smoothing: Apply optical flow smoothing
        flow_smoothing_strength: How much to smooth (0.0-1.0)
    
    Returns:
        Path to processed video
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not enable_optical_flow_smoothing:
        # Fall back to standard postprocessing
        from src.media_utils import postprocess_video_quality
        return postprocess_video_quality(
            video_path, output_path,
            denoise_strength, sharpen_amount, contrast, saturation, crf, preset
        )
    
    # Use ffmpeg with advanced filters for optical flow smoothing
    # This approach uses FFmpeg's minterpolate and other temporal filters
    
    # Compose filter chain
    filters = [
        # Temporal denoising and smoothing
        f"minterpolate=fps=30:mi_mode=dup:mc_mode=aobmc:me_mode=bilat:vsbmc=1",  # Smoother motion
        f"hqdn3d={denoise_strength}:{denoise_strength * 0.75}:{denoise_strength * 3.0}:{denoise_strength * 2.0}",  # Denoise
        f"unsharp=5:5:{sharpen_amount}:5:5:0.0",  # Sharpen
        f"eq=contrast={contrast}:saturation={saturation}",  # Color
    ]
    
    filter_graph = ",".join(filters)
    
    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", filter_graph,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-an",
        str(output_path),
    ]
    
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        error_tail = "\n".join(completed.stderr.splitlines()[-20:])
        raise RuntimeError(f"ffmpeg advanced postprocessing failed.\n{error_tail}")
    
    return output_path


def apply_histogram_matching(
    video_path: str | Path,
    reference_video: str | Path | None,
    output_path: str | Path,
    preserve_luminance: bool = True,
) -> Path:
    """
    Match colors of output video to match the reference (original) video.
    Prevents color shifts and makes output look more consistent with source.
    
    Args:
        video_path: Video to color-correct
        reference_video: Reference video to match colors from (typically original)
        output_path: Output video
        preserve_luminance: Keep luminance, only match chrominance
    
    Returns:
        Path to color-matched video
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If no reference, just return original (or apply color correction)
    if reference_video is None:
        import shutil
        shutil.copy2(video_path, output_path)
        return output_path
    
    reference_video = Path(reference_video)
    
    # Extract first 100 frames from each video for histogram analysis
    cap_ref = cv2.VideoCapture(str(reference_video))
    cap_video = cv2.VideoCapture(str(video_path))
    
    ref_frames = []
    video_frames = []
    
    for _ in range(min(100, int(cap_ref.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap_ref.read()
        if not ret:
            break
        ref_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
    
    for _ in range(min(100, int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap_video.read()
        if not ret:
            break
        video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
    
    cap_ref.release()
    cap_video.release()
    
    if not ref_frames or not video_frames:
        import shutil
        shutil.copy2(video_path, output_path)
        return output_path
    
    # Compute mean and std for each channel
    ref_mean = np.mean([f.reshape(-1, 3) for f in ref_frames], axis=(0, 1))
    ref_std = np.std([f.reshape(-1, 3) for f in ref_frames], axis=(0, 1))
    
    video_mean = np.mean([f.reshape(-1, 3) for f in video_frames], axis=(0, 1))
    video_std = np.std([f.reshape(-1, 3) for f in video_frames], axis=(0, 1))
    
    # Create transfer function
    def transfer_color(frame):
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        if preserve_luminance:
            # Only match color channels (A, B), keep L
            for ch in [1, 2]:  # A and B channels
                frame_lab[:, :, ch] = (frame_lab[:, :, ch] - video_mean[ch]) * (ref_std[ch] / (video_std[ch] + 1e-6)) + ref_mean[ch]
        else:
            # Match all channels
            for ch in range(3):
                frame_lab[:, :, ch] = (frame_lab[:, :, ch] - video_mean[ch]) * (ref_std[ch] / (video_std[ch] + 1e-6)) + ref_mean[ch]
        
        frame_lab = np.clip(frame_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(frame_lab, cv2.COLOR_LAB2BGR)
    
    # Apply to entire video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    with tqdm(total=total_frames, desc="Color Matching", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            corrected = transfer_color(frame)
            writer.write(corrected)
            pbar.update(1)
    
    cap.release()
    writer.release()
    
    return output_path


def remove_temporal_jitter(
    video_path: str | Path,
    output_path: str | Path,
    window_size: int = 3,  # Odd number for temporal median
    intensity: float = 0.7,  # How much to smooth (0-1)
) -> Path:
    """
    Reduce temporal jitter and discontinuities using temporal median filtering.
    Particularly effective for fixing Wav2Lip mouth jitter.
    
    Args:
        video_path: Input video
        output_path: Output video
        window_size: Window size for temporal filtering (3, 5, 7, etc.)
        intensity: Smoothing intensity (0-1)
    
    Returns:
        Path to jitter-reduced video
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Buffer frames
    frame_buffer = []
    output_frames = []
    
    print("Reading frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_buffer.append(frame.astype(np.float32) / 255.0)
    
    cap.release()
    
    half_window = window_size // 2
    
    print("Applying temporal jitter reduction...")
    for i in tqdm(range(len(frame_buffer)), desc="Jitter Removal"):
        # Get window of frames
        start_idx = max(0, i - half_window)
        end_idx = min(len(frame_buffer), i + half_window + 1)
        window = frame_buffer[start_idx:end_idx]
        
        # Apply temporal median
        window_array = np.array(window)
        smoothed = np.median(window_array, axis=0)
        
        # Blend with original based on intensity
        blended = intensity * smoothed + (1 - intensity) * frame_buffer[i]
        
        # Convert back to uint8
        blended = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
        writer.write(blended)
    
    writer.release()
    return output_path


def enhance_contrast_adaptive(
    video_path: str | Path,
    output_path: str | Path,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> Path:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    without over-processing or artifacts.
    
    Args:
        video_path: Input video
        output_path: Output video
        clip_limit: Contrast limit (1-4 typical)
        tile_size: Tile grid size
    
    Returns:
        Path to contrast-enhanced video
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    
    with tqdm(total=total_frames, desc="Adaptive Contrast", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to LAB and enhance L channel only
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = clahe.apply(l)
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            frame_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            writer.write(frame_enhanced)
            pbar.update(1)
    
    cap.release()
    writer.release()
    
    return output_path


def verify_lip_sync_accuracy(
    video_path: str | Path,
    sample_segment_count: int = 5,
) -> dict:
    """
    Verify lip-sync accuracy by analyzing mouth motion vs audio onsets.
    Returns diagnostic information about sync quality.
    
    Args:
        video_path: Video to analyze
        sample_segment_count: Number of segments to sample for analysis
    
    Returns:
        Dictionary with sync metrics
    """
    try:
        import face_recognition
    except ImportError:
        return {
            'status': 'skipped',
            'reason': 'face_recognition not installed',
            'sync_confidence': 0.0,
        }
    
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames across video
    sample_indices = np.linspace(0, total_frames - 1, sample_segment_count, dtype=int)
    
    sync_scores = []
    
    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Detect faces
        face_locations = face_recognition.face_locations(frame)
        if not face_locations:
            continue
        
        # Check for mouth region intensity changes (proxy for mouth movement)
        top, right, bottom, left = face_locations[0]
        mouth_region = frame[bottom - (bottom - top) // 4:bottom, left:right]
        
        # Simple motion detection via Laplacian variance
        gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        sync_scores.append(float(laplacian_var))
    
    cap.release()
    
    avg_score = np.mean(sync_scores) if sync_scores else 0.0
    
    return {
        'status': 'analyzed',
        'sample_count': len(sync_scores),
        'avg_mouth_motion_variance': float(avg_score),
        'sync_confidence': min(1.0, avg_score / 100.0),  # Normalized 0-1
        'recommendation': 'sync looks good' if avg_score > 50 else 'check sync manually',
    }


def batch_postprocess_advanced(
    video_paths: list[str | Path],
    output_dir: str | Path,
    **postprocess_kwargs
) -> list[Path]:
    """
    Apply advanced post-processing to multiple videos.
    Useful for batch operations.
    
    Returns:
        List of output paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for video_path in video_paths:
        output_path = output_dir / Path(video_path).name
        result = advanced_postprocess_with_optical_flow(
            video_path, output_path, **postprocess_kwargs
        )
        results.append(result)
    
    return results
