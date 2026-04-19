from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import librosa
import numpy as np
import soundfile as sf

try:
    from moviepy import VideoFileClip
except ImportError:  # pragma: no cover - compatibility for older MoviePy versions
    from moviepy.editor import VideoFileClip


def ensure_ffmpeg_available() -> None:
    """Raise a helpful error if ffmpeg is not on PATH."""

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg was not found on PATH. Install ffmpeg and make sure the binary is available before running the pipeline."
        )


def ensure_cuda_available() -> None:
    """Raise a helpful error if CUDA is not available for the model stack."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "PyTorch is required to check CUDA availability and run Whisper/XTTS/Wav2Lip. Install torch with CUDA support."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This pipeline is configured to require a GPU for Whisper, XTTS v2, and Wav2Lip."
        )


def resolve_runtime_device(requested_device: str = "auto", require_cuda: bool = False) -> str:
    """Resolve runtime device from user preference and CUDA availability.

    Returns "cuda" or "cpu".
    """

    requested_device = requested_device.lower().strip()
    if requested_device not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Invalid device '{requested_device}'. Use one of: auto, cuda, cpu.")

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        if requested_device == "cuda" or require_cuda:
            raise RuntimeError(
                "PyTorch is required to use CUDA but is not installed. Install torch with CUDA support."
            ) from exc
        return "cpu"

    cuda_available = torch.cuda.is_available()

    if require_cuda and not cuda_available:
        raise RuntimeError(
            "CUDA is not available but --require_cuda was specified. "
            "Install GPU drivers/CUDA runtime or rerun without --require_cuda."
        )

    if requested_device == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA device was explicitly requested but is not available. "
            "Use --device auto or --device cpu."
        )

    if requested_device == "cpu":
        return "cpu"
    if requested_device == "cuda":
        return "cuda"

    return "cuda" if cuda_available else "cpu"


def get_media_duration(media_path: str | Path) -> float:
    """Return media duration in seconds using MoviePy."""

    media_path = Path(media_path)
    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    suffix = media_path.suffix.lower()
    if suffix in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}:
        return float(librosa.get_duration(path=str(media_path)))

    clip = VideoFileClip(str(media_path))
    try:
        return float(clip.duration)
    finally:
        clip.close()


def extract_audio_from_video(video_path: str | Path, output_audio_path: str | Path, sample_rate: int = 22050) -> Path:
    """Extract the audio track from a video file into a WAV file."""

    video_path = Path(video_path)
    output_audio_path = Path(output_audio_path)
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    clip = VideoFileClip(str(video_path))
    try:
        if clip.audio is None:
            raise RuntimeError(f"No audio track found in video: {video_path}")
        write_kwargs = {
            "fps": sample_rate,
            "codec": "pcm_s16le",
            "verbose": False,
            "logger": None,
        }
        try:
            clip.audio.write_audiofile(str(output_audio_path), **write_kwargs)
        except TypeError as exc:
            # MoviePy versions differ in accepted logging kwargs.
            if "unexpected keyword argument" not in str(exc):
                raise
            write_kwargs.pop("verbose", None)
            write_kwargs.pop("logger", None)
            clip.audio.write_audiofile(str(output_audio_path), **write_kwargs)
    finally:
        clip.close()

    return output_audio_path


def extract_voice_sample(audio_path: str | Path, output_sample_path: str | Path, sample_seconds: float = 10.0) -> Path:
    """Extract an initial voice reference sample for XTTS speaker cloning."""

    audio_path = Path(audio_path)
    output_sample_path = Path(output_sample_path)
    output_sample_path.parent.mkdir(parents=True, exist_ok=True)

    waveform, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
    sample_length = min(len(waveform), int(round(sample_seconds * sample_rate)))
    if sample_length <= 0:
        raise RuntimeError(f"Unable to extract a voice sample from empty audio: {audio_path}")

    sf.write(str(output_sample_path), waveform[:sample_length], sample_rate)
    return output_sample_path


def calculate_stretch_factor(video_path: str | Path, audio_path: str | Path) -> tuple[float, float, float]:
    """Return the source audio duration, video duration, and stretch factor.

    The stretch factor is calculated for librosa.effects.time_stretch as:
    source_audio_duration / video_duration
    """

    video_duration = get_media_duration(video_path)
    audio_duration = get_media_duration(audio_path)

    if video_duration <= 0:
        raise ValueError(f"Invalid video duration for {video_path}: {video_duration}")
    if audio_duration <= 0:
        raise ValueError(f"Invalid audio duration for {audio_path}: {audio_duration}")

    stretch_factor = audio_duration / video_duration
    return audio_duration, video_duration, stretch_factor


def stretch_audio_to_video_duration(
    video_path: str | Path,
    audio_path: str | Path,
    output_audio_path: str | Path,
) -> tuple[Path, float]:
    """Time-stretch audio so it matches the video duration without changing pitch."""

    audio_path = Path(audio_path)
    output_audio_path = Path(output_audio_path)
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    _source_duration, target_duration, stretch_factor = calculate_stretch_factor(video_path, audio_path)
    waveform, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)

    if waveform.size == 0:
        raise RuntimeError(f"Generated audio is empty: {audio_path}")

    stretched_waveform = librosa.effects.time_stretch(waveform, rate=stretch_factor)
    target_samples = int(round(target_duration * sample_rate))

    if stretched_waveform.shape[0] > target_samples:
        stretched_waveform = stretched_waveform[:target_samples]
    elif stretched_waveform.shape[0] < target_samples:
        stretched_waveform = np.pad(stretched_waveform, (0, target_samples - stretched_waveform.shape[0]))

    sf.write(str(output_audio_path), stretched_waveform, sample_rate)
    return output_audio_path, stretch_factor


def pad_or_trim_audio_to_video_duration(
    video_path: str | Path,
    audio_path: str | Path,
    output_audio_path: str | Path,
) -> Path:
    """Pad or trim audio to match video duration exactly, preserving local timing."""

    audio_path = Path(audio_path)
    output_audio_path = Path(output_audio_path)
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    video_duration = get_media_duration(video_path)
    if video_duration <= 0:
        raise ValueError(f"Invalid video duration for {video_path}: {video_duration}")

    waveform, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
    if waveform.size == 0:
        raise RuntimeError(f"Generated audio is empty: {audio_path}")

    target_samples = int(round(video_duration * sample_rate))
    if waveform.shape[0] > target_samples:
        waveform = waveform[:target_samples]
    elif waveform.shape[0] < target_samples:
        waveform = np.pad(waveform, (0, target_samples - waveform.shape[0]))

    sf.write(str(output_audio_path), waveform, sample_rate)
    return output_audio_path


def postprocess_video_quality(
    video_path: str | Path,
    output_video_path: str | Path,
    denoise_strength: float = 1.2,
    sharpen_amount: float = 0.6,
    contrast: float = 1.02,
    saturation: float = 1.03,
    crf: int = 16,
    preset: str = "slow",
) -> Path:
    """Apply mild denoise/sharpen/color correction to improve perceived output quality."""

    video_path = Path(video_path)
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found for postprocessing: {video_path}")
    if denoise_strength < 0:
        raise ValueError(f"denoise_strength must be >= 0, got {denoise_strength}")
    if sharpen_amount < 0:
        raise ValueError(f"sharpen_amount must be >= 0, got {sharpen_amount}")
    if contrast <= 0:
        raise ValueError(f"contrast must be > 0, got {contrast}")
    if saturation <= 0:
        raise ValueError(f"saturation must be > 0, got {saturation}")
    if not 0 <= crf <= 51:
        raise ValueError(f"crf must be in [0, 51], got {crf}")

    allowed_presets = {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    }
    if preset not in allowed_presets:
        raise ValueError(f"Invalid preset '{preset}'. Choose one of: {', '.join(sorted(allowed_presets))}")

    luma_temporal = max(1.0, denoise_strength * 3.0)
    chroma_temporal = max(1.0, denoise_strength * 2.0)
    video_filter = (
        f"hqdn3d={denoise_strength}:{denoise_strength * 0.75}:{luma_temporal}:{chroma_temporal},"
        f"unsharp=5:5:{sharpen_amount}:5:5:0.0,"
        f"eq=contrast={contrast}:saturation={saturation}"
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output_video_path),
    ]

    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        error_tail = "\n".join(completed.stderr.splitlines()[-20:])
        raise RuntimeError(f"ffmpeg postprocessing failed.\n{error_tail}")

    return output_video_path


def mux_video_with_audio(video_path: str | Path, audio_path: str | Path, output_video_path: str | Path) -> Path:
    """Attach audio to video and write the final muxed file."""

    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Prefer stream-copy to avoid re-encoding Wav2Lip output and losing facial detail.
    command_copy = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(output_video_path),
    ]
    copy_run = subprocess.run(command_copy, capture_output=True, text=True)
    if copy_run.returncode != 0:
        command_reencode = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(output_video_path),
        ]
        reencode_run = subprocess.run(command_reencode, capture_output=True, text=True)
        if reencode_run.returncode != 0:
            error_tail = "\n".join(reencode_run.stderr.splitlines()[-20:])
            raise RuntimeError(f"ffmpeg mux failed.\n{error_tail}")

    return output_video_path
