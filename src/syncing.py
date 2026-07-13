from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import sys

from tqdm import tqdm


_PROGRESS_PATTERN = re.compile(r"(\d{1,3}(?:\.\d+)?)%")


def run_wav2lip_inference(
    checkpoint_path: str | Path,
    face_video_path: str | Path,
    audio_path: str | Path,
    output_video_path: str | Path,
    wav2lip_root: str | Path = "Wav2Lip",
    python_executable: str = sys.executable,
    pads: tuple[int, int, int, int] = (0, 20, 0, 0),
    resize_factor: int = 1,
    no_smooth: bool = False,
    face_det_batch_size: int = 16,
    wav2lip_batch_size: int = 128,
    crop: tuple[int, int, int, int] = (0, -1, 0, -1),
    rotate: bool = False,
    box: tuple[int, int, int, int] | None = None,
    exclude_intervals_path: str | Path | None = None,
) -> Path:
    """Run Wav2Lip as a subprocess and stream its output with progress feedback."""

    checkpoint_path = Path(checkpoint_path)
    face_video_path = Path(face_video_path)
    audio_path = Path(audio_path)
    output_video_path = Path(output_video_path)
    wav2lip_root = Path(wav2lip_root)
    inference_script = wav2lip_root / "inference.py"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Wav2Lip checkpoint not found: {checkpoint_path}")
    if not face_video_path.exists():
        raise FileNotFoundError(f"Face video not found: {face_video_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not inference_script.exists():
        raise FileNotFoundError(
            f"Wav2Lip inference script not found: {inference_script}. "
            "Clone the repository into the project root (example: git clone https://github.com/Rudrabha/Wav2Lip.git) "
            "or pass the correct path via --wav2lip_root."
        )
    if resize_factor < 1:
        raise ValueError(f"resize_factor must be >= 1, got {resize_factor}")
    if face_det_batch_size < 1:
        raise ValueError(f"face_det_batch_size must be >= 1, got {face_det_batch_size}")
    if wav2lip_batch_size < 1:
        raise ValueError(f"wav2lip_batch_size must be >= 1, got {wav2lip_batch_size}")

    checkpoint_path = checkpoint_path.resolve()
    face_video_path = face_video_path.resolve()
    audio_path = audio_path.resolve()
    output_video_path = output_video_path.resolve()
    wav2lip_root = wav2lip_root.resolve()
    inference_script = inference_script.resolve()

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    wav2lip_temp_output = wav2lip_root / "wav2lip_out.mp4"

    command = [
        python_executable,
        inference_script.name,
        "--checkpoint_path",
        str(checkpoint_path),
        "--face",
        str(face_video_path),
        "--audio",
        str(audio_path),
        "--outfile",
        str(wav2lip_temp_output),
        "--pads",
        str(pads[0]),
        str(pads[1]),
        str(pads[2]),
        str(pads[3]),
        "--resize_factor",
        str(resize_factor),
        "--face_det_batch_size",
        str(face_det_batch_size),
        "--wav2lip_batch_size",
        str(wav2lip_batch_size),
        "--crop",
        str(crop[0]),
        str(crop[1]),
        str(crop[2]),
        str(crop[3]),
    ]

    if no_smooth:
        command.append("--nosmooth")
    if rotate:
        command.append("--rotate")
    if box is not None:
        command.extend([
            "--box",
            str(box[0]),
            str(box[1]),
            str(box[2]),
            str(box[3]),
        ])
    if exclude_intervals_path is not None:
        command.extend([
            "--exclude_intervals",
            str(exclude_intervals_path),
        ])

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(wav2lip_root),
    )

    if process.stdout is None:
        raise RuntimeError("Failed to capture Wav2Lip stdout.")

    captured_lines: list[str] = []
    progress_value = 0.0

    with tqdm(total=100, desc="Wav2Lip", unit="%") as progress_bar:
        for raw_line in iter(process.stdout.readline, ""):
            line = raw_line.rstrip()
            if not line:
                continue

            captured_lines.append(line)
            print(line, flush=True)

            match = _PROGRESS_PATTERN.search(line)
            if match:
                new_value = min(float(match.group(1)), 100.0)
                delta = new_value - progress_value
                if delta > 0:
                    progress_bar.update(delta)
                    progress_value = new_value

        process.stdout.close()
        return_code = process.wait()

        if return_code != 0:
            tail = "\n".join(captured_lines[-20:])
            raise RuntimeError(f"Wav2Lip failed with exit code {return_code}.\n{tail}")

        if not wav2lip_temp_output.exists():
            raise FileNotFoundError(
                f"Wav2Lip output file was not created: {wav2lip_temp_output}. "
                "This may be due to path quoting issues or internal Wav2Lip errors."
            )

        shutil.move(str(wav2lip_temp_output), str(output_video_path))

        if progress_value < 100.0:
            progress_bar.update(100.0 - progress_value)

    return output_video_path
