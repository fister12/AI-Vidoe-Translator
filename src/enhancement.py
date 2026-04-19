from __future__ import annotations

from pathlib import Path
import sys
import types


def _ensure_torchvision_functional_tensor_compat() -> None:
    """Provide a compatibility shim for packages expecting torchvision.functional_tensor."""

    module_name = "torchvision.transforms.functional_tensor"
    if module_name in sys.modules:
        return

    try:
        from torchvision.transforms import functional as transforms_functional
    except Exception:
        return

    shim_module = types.ModuleType(module_name)
    if hasattr(transforms_functional, "rgb_to_grayscale"):
        shim_module.rgb_to_grayscale = transforms_functional.rgb_to_grayscale

    sys.modules[module_name] = shim_module

def enhance_faces_in_video(
    video_path: str | Path,
    output_video_path: str | Path,
    device: str = "cpu",
) -> Path:
    """Enhance faces in a video using GFPGAN."""
    _ensure_torchvision_functional_tensor_compat()
    try:
        from gfpgan import GFPGANer
    except ImportError as exc:
        raise RuntimeError(
            "gfpgan could not be imported. This is often caused by torchvision/basicsr "
            "compatibility issues in the current environment. "
            "Try running without --enhance, or align gfpgan/basicsr/torchvision versions."
        ) from exc

    import cv2
    from tqdm import tqdm

    video_path = Path(video_path)
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    model_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"

    print("Initializing GFPGAN face restorer...")
    restorer = GFPGANer(
        model_path=model_url,
        upscale=1,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None,
        device=device
    )

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video writer for {output_video_path}")

    try:
        with tqdm(total=total_frames, desc="GFPGAN Enhancement", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Enhance face
                _, _, restored_img = restorer.enhance(
                    frame,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                    weight=0.5
                )
                
                if restored_img is not None:
                    writer.write(restored_img)
                else:
                    writer.write(frame)
                    
                pbar.update(1)
    finally:
        cap.release()
        writer.release()
        
    return output_video_path
