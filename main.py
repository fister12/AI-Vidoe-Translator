from __future__ import annotations

import argparse
from pathlib import Path

from src.media_utils import (
    ensure_ffmpeg_available,
    extract_audio_from_video,
    extract_voice_sample,
    mux_video_with_audio,
    pad_or_trim_audio_to_video_duration,
    postprocess_video_quality,
    resolve_runtime_device,
    stretch_audio_to_video_duration,
)
from src.enhancement import enhance_faces_in_video
from src.syncing import run_wav2lip_inference
from src.transcription import transcribe_english_audio
from src.translation import translate_segments, translate_text
from src.tts import synthesize_aligned_audio_from_segments, synthesize_spanish_audio


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Language-To-Language dub and sync pipeline.")
    parser.add_argument("--input_video", required=True, help="Path to the source Language video.")
    parser.add_argument("--output_video", required=True, help="Path for the final dubbed video.")
    parser.add_argument("--checkpoint_path", required=True, help="Path to the Wav2Lip checkpoint file.")
    parser.add_argument("--wav2lip_root", default="Wav2Lip", help="Path to the Wav2Lip repository root.")
    parser.add_argument("--whisper_model", default="small", help="Whisper model size, e.g. tiny, base, small, medium, large.")
    parser.add_argument(
        "--audio_language",
        "--Initial_language",
        dest="audio_language",
        default="en",
        help="Language code for the input audio. Use 'auto' to let Whisper detect it.",
    )
    parser.add_argument(
        "--output_language",
        "--target_language",
        dest="output_language",
        default="es",
        help="Language code for the translated output audio.",
    )
    parser.add_argument(
        "--tts_model_name",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="Coqui XTTS model name.",
    )
    parser.add_argument("--working_dir", default="temp", help="Directory for intermediate artifacts.")
    parser.add_argument(
        "--timing_mode",
        choices=["segment", "global"],
        default="segment",
        help="Audio timing strategy: segment preserves per-utterance timing, global stretches one full track.",
    )
    parser.add_argument("--extract_sample_seconds", type=float, default=10.0, help="Length of voice sample for cloning.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for Whisper/XTTS: auto chooses CUDA when available, else CPU.",
    )
    parser.add_argument(
        "--require_cuda",
        action="store_true",
        help="Fail fast if CUDA is not available.",
    )
    parser.add_argument(
        "--enhance",
        action="store_true",
        help="Apply GFPGAN face enhancement pass after Wav2Lip.",
    )
    parser.add_argument(
        "--postprocess_denoise_strength",
        type=float,
        default=1.2,
        help="Postprocess denoise strength (higher removes more compression noise).",
    )
    parser.add_argument(
        "--postprocess_sharpen_amount",
        type=float,
        default=0.6,
        help="Postprocess sharpening amount.",
    )
    parser.add_argument(
        "--postprocess_contrast",
        type=float,
        default=1.02,
        help="Postprocess contrast multiplier.",
    )
    parser.add_argument(
        "--postprocess_saturation",
        type=float,
        default=1.03,
        help="Postprocess saturation multiplier.",
    )
    parser.add_argument(
        "--postprocess_crf",
        type=int,
        default=16,
        help="x264 CRF for postprocessed video (lower is higher quality).",
    )
    parser.add_argument(
        "--postprocess_preset",
        default="slow",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        help="x264 preset used for postprocess encoding.",
    )
    parser.add_argument(
        "--wav2lip_pads",
        nargs=4,
        type=int,
        metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"),
        default=[0, 20, 0, 0],
        help="Face padding passed to Wav2Lip (top bottom left right). Increase bottom to include chin.",
    )
    parser.add_argument(
        "--wav2lip_resize_factor",
        type=int,
        default=1,
        help="Downscale input frames by this factor before Wav2Lip face detection.",
    )
    parser.add_argument(
        "--wav2lip_no_smooth",
        action="store_true",
        help="Disable smoothing of face detections (helps when mouth looks duplicated/dislocated).",
    )
    parser.add_argument(
        "--wav2lip_face_det_batch_size",
        type=int,
        default=16,
        help="Face detection batch size for Wav2Lip.",
    )
    parser.add_argument(
        "--wav2lip_batch_size",
        type=int,
        default=128,
        help="Inference batch size for Wav2Lip model.",
    )
    parser.add_argument(
        "--wav2lip_crop",
        nargs=4,
        type=int,
        metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"),
        default=[0, -1, 0, -1],
        help="Optional crop region passed to Wav2Lip (top bottom left right).",
    )
    parser.add_argument(
        "--wav2lip_rotate",
        action="store_true",
        help="Rotate frames 90 degrees clockwise before Wav2Lip (for wrongly oriented phone videos).",
    )
    parser.add_argument(
        "--wav2lip_box",
        nargs=4,
        type=int,
        metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"),
        default=None,
        help="Optional fixed face box for Wav2Lip (use only if detection is unstable).",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    args.audio_language = str(args.audio_language).strip().lower()
    args.output_language = str(args.output_language).strip().lower()

    if not args.audio_language:
        parser.error("--audio_language cannot be empty.")
    if not args.output_language:
        parser.error("--output_language cannot be empty.")
    if args.output_language == "auto":
        parser.error("--output_language cannot be 'auto'. Provide a target language code such as 'es'.")

    ensure_ffmpeg_available()
    device = resolve_runtime_device(requested_device=args.device, require_cuda=args.require_cuda)
    print(f"Using runtime device: {device}")

    input_video = Path(args.input_video)
    output_video = Path(args.output_video)
    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    extracted_audio_path = working_dir / "english_audio.wav"
    voice_sample_path = working_dir / "speaker_sample.wav"
    translated_audio_raw_path = working_dir / "spanish_audio_raw.wav"
    translated_audio_synced_path = working_dir / "spanish_audio_synced.wav"
    wav2lip_output_path = working_dir / "wav2lip_output.mp4"
    postprocessed_video_path = working_dir / "wav2lip_postprocessed.mp4"

    extract_audio_from_video(input_video, extracted_audio_path)
    whisper_language = None if args.audio_language == "auto" else args.audio_language
    transcript_text, transcript_segments, detected_audio_language = transcribe_english_audio(
        extracted_audio_path,
        model_size=args.whisper_model,
        device=device,
        language=whisper_language,
    )

    translation_source_language = args.audio_language
    if translation_source_language == "auto":
        translation_source_language = detected_audio_language or "en"

    print(f"Resolved source language: {translation_source_language}")
    print(f"Resolved target language: {args.output_language}")

    extract_voice_sample(
        extracted_audio_path,
        voice_sample_path,
        sample_seconds=args.extract_sample_seconds,
    )

    used_segment_timing = False
    if args.timing_mode == "segment" and transcript_segments:
        try:
            translated_segments = translate_segments(
                transcript_segments,
                source_language=translation_source_language,
                target_language=args.output_language,
            )
            synthesize_aligned_audio_from_segments(
                segments=translated_segments,
                speaker_wav_path=voice_sample_path,
                output_audio_path=translated_audio_raw_path,
                model_name=args.tts_model_name,
                language=args.output_language,
                gpu=device == "cuda",
            )
            pad_or_trim_audio_to_video_duration(
                video_path=input_video,
                audio_path=translated_audio_raw_path,
                output_audio_path=translated_audio_synced_path,
            )
            used_segment_timing = True
            print("Timing mode: segment-aligned")
        except Exception as exc:
            print(f"Segment timing failed ({exc}); falling back to global timing.")

    if not used_segment_timing:
        translated_text = translate_text(
            transcript_text,
            source_language=translation_source_language,
            target_language=args.output_language,
        )
        synthesize_spanish_audio(
            text=translated_text,
            speaker_wav_path=voice_sample_path,
            output_audio_path=translated_audio_raw_path,
            model_name=args.tts_model_name,
            language=args.output_language,
            gpu=device == "cuda",
        )

        stretch_audio_to_video_duration(
            video_path=input_video,
            audio_path=translated_audio_raw_path,
            output_audio_path=translated_audio_synced_path,
        )
        print("Timing mode: global-stretch")

    run_wav2lip_inference(
        checkpoint_path=args.checkpoint_path,
        face_video_path=input_video,
        audio_path=translated_audio_synced_path,
        output_video_path=wav2lip_output_path,
        wav2lip_root=args.wav2lip_root,
        pads=tuple(args.wav2lip_pads),
        resize_factor=args.wav2lip_resize_factor,
        no_smooth=args.wav2lip_no_smooth,
        face_det_batch_size=args.wav2lip_face_det_batch_size,
        wav2lip_batch_size=args.wav2lip_batch_size,
        crop=tuple(args.wav2lip_crop),
        rotate=args.wav2lip_rotate,
        box=tuple(args.wav2lip_box) if args.wav2lip_box is not None else None,
    )

    video_for_postprocess = wav2lip_output_path
    if args.enhance:
        enhanced_temp_path = working_dir / "wav2lip_enhanced.mp4"
        enhance_faces_in_video(wav2lip_output_path, enhanced_temp_path, device=device)
        video_for_postprocess = enhanced_temp_path

    postprocess_video_quality(
        video_path=video_for_postprocess,
        output_video_path=postprocessed_video_path,
        denoise_strength=args.postprocess_denoise_strength,
        sharpen_amount=args.postprocess_sharpen_amount,
        contrast=args.postprocess_contrast,
        saturation=args.postprocess_saturation,
        crf=args.postprocess_crf,
        preset=args.postprocess_preset,
    )

    mux_video_with_audio(
        video_path=postprocessed_video_path,
        audio_path=translated_audio_synced_path,
        output_video_path=output_video,
    )

    print(f"Final output written to: {output_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
