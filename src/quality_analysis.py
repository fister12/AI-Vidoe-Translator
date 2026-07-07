from __future__ import annotations

from pathlib import Path
import numpy as np
import librosa
import cv2
from tqdm import tqdm


def optimize_transcription_segments(
    segments: list[dict],
    min_confidence: float = 0.80,
    max_segment_duration: float = 5.0,
    min_segment_duration: float = 0.5,
    merge_nearby_pauses: bool = True,
    pause_threshold_ms: float = 500.0,
) -> list[dict]:
    """
    Post-process Whisper segments for better TTS quality and sync.
    
    Improvements:
    1. Filter low-confidence segments
    2. Split long segments at natural breaks
    3. Merge short segments
    4. Remove excessive pauses between segments
    
    Args:
        segments: Whisper transcription segments
        min_confidence: Minimum confidence to keep segment
        max_segment_duration: Split segments longer than this
        min_segment_duration: Merge segments shorter than this
        merge_nearby_pauses: Merge segments separated by small pauses
        pause_threshold_ms: Max pause between segments to merge (ms)
    
    Returns:
        Optimized segments list
    """
    if not segments:
        return []
    
    # Step 1: Filter by confidence
    filtered = []
    for seg in segments:
        confidence = float(seg.get('confidence', 1.0))
        if confidence >= min_confidence:
            filtered.append(seg)
    
    if not filtered:
        return segments  # Return original if filtering removed everything
    
    # Step 2: Split long segments at sentence boundaries
    split_segments = []
    for seg in filtered:
        text = str(seg.get('text', '')).strip()
        start = float(seg.get('start', 0.0))
        end = float(seg.get('end', start))
        duration = end - start
        
        if duration > max_segment_duration:
            # Split by sentences
            sentences = text.split('.')
            if len(sentences) > 1:
                # Estimate character-based duration
                char_per_second = len(text) / duration
                seg_start = start
                
                for i, sentence in enumerate(sentences[:-1]):
                    sentence = sentence.strip() + '.'
                    if sentence.strip():
                        est_duration = len(sentence) / char_per_second
                        split_segments.append({
                            'start': seg_start,
                            'end': seg_start + est_duration,
                            'text': sentence,
                            'confidence': seg.get('confidence', 1.0),
                        })
                        seg_start += est_duration
                
                # Last sentence
                last = sentences[-1].strip()
                if last:
                    split_segments.append({
                        'start': seg_start,
                        'end': end,
                        'text': last,
                        'confidence': seg.get('confidence', 1.0),
                    })
            else:
                split_segments.append(seg)
        else:
            split_segments.append(seg)
    
    # Step 3: Merge short segments with adjacent ones
    merged = []
    i = 0
    while i < len(split_segments):
        current = dict(split_segments[i])
        duration = current['end'] - current['start']
        
        if duration < min_segment_duration and i < len(split_segments) - 1:
            # Merge with next segment
            next_seg = split_segments[i + 1]
            current['text'] = current['text'].strip() + ' ' + str(next_seg.get('text', '')).strip()
            current['end'] = next_seg['end']
            current['confidence'] = min(float(current.get('confidence', 1.0)), float(next_seg.get('confidence', 1.0)))
            i += 1
        
        merged.append(current)
        i += 1
    
    # Step 4: Remove excessive pauses
    if merge_nearby_pauses:
        pause_threshold_sec = pause_threshold_ms / 1000.0
        final = []
        i = 0
        while i < len(merged):
            current = dict(merged[i])
            
            if i < len(merged) - 1:
                next_seg = merged[i + 1]
                pause = next_seg['start'] - current['end']
                
                if pause < pause_threshold_sec:
                    # Merge by adding silence indicator
                    if not current['text'].endswith(' '):
                        current['text'] += ' '
                    current['text'] += str(next_seg.get('text', '')).strip()
                    current['end'] = next_seg['end']
                    i += 1
            
            final.append(current)
            i += 1
        
        return final
    
    return merged


def analyze_segment_pronunciation_difficulty(
    segments: list[dict],
    target_language: str,
) -> list[dict]:
    """
    Analyze segments for pronunciation difficulty and add metadata.
    Helps identify which segments might benefit from special handling.
    
    Args:
        segments: Translation segments
        target_language: Target language code
    
    Returns:
        Segments with added 'difficulty' and 'phonemes' fields
    """
    try:
        import textstat
        from epitran import Epitran
    except ImportError:
        # Return original if dependencies missing
        return segments
    
    # Map language codes to epitran codes
    lang_map = {
        'en': 'eng-Latn',
        'es': 'spa-Latn',
        'fr': 'fra-Latn',
        'de': 'deu-Latn',
        'it': 'ita-Latn',
        'pt': 'por-Latn',
        'ja': 'jpn',
        'zh': 'cmn-Hans',
        'hi': 'hin-Deva',
    }
    
    lang_code = lang_map.get(target_language[:2].lower())
    if not lang_code:
        return segments
    
    try:
        epitran = Epitran(lang_code)
    except Exception:
        return segments
    
    enhanced = []
    for seg in segments:
        text = str(seg.get('text', '')).strip()
        
        # Compute difficulty metrics
        flesch_kincaid = textstat.flesch_kincaid_grade(text) if text else 0
        syllable_count = textstat.syllable_count(text) if text else 0
        
        # Get phonetic transcription
        try:
            phonemes = epitran.transliterate(text)
        except:
            phonemes = text
        
        enhanced.append({
            **seg,
            'difficulty_grade': float(flesch_kincaid),
            'syllable_count': int(syllable_count),
            'phonemes': phonemes,
            'is_difficult': flesch_kincaid > 12,
        })
    
    return enhanced


def estimate_optimal_tts_parameters(
    segments: list[dict],
) -> dict:
    """
    Estimate optimal TTS parameters based on segment analysis.
    
    Returns:
        Dictionary with recommended parameters
    """
    if not segments:
        return {}
    
    # Analyze segment statistics
    durations = [float(seg.get('end', 0)) - float(seg.get('start', 0)) for seg in segments]
    text_lengths = [len(str(seg.get('text', ''))) for seg in segments]
    
    avg_duration = np.mean(durations) if durations else 1.0
    avg_text_length = np.mean(text_lengths) if text_lengths else 10
    
    # Speech rate in chars per second
    speech_rate = avg_text_length / avg_duration if avg_duration > 0 else 10
    
    return {
        'avg_segment_duration': float(avg_duration),
        'avg_text_length': float(avg_text_length),
        'estimated_speech_rate': float(speech_rate),
        'recommended_max_segment_duration': min(5.0, float(avg_duration * 3)),
        'recommended_batch_size': max(1, min(8, int(30 / (avg_duration + 0.1)))),
    }


class VideoQualityScorer:
    """Comprehensive video quality assessment."""
    
    @staticmethod
    def compute_blur_score(frame: np.ndarray) -> float:
        """
        Compute blur score using Laplacian variance.
        Higher = sharper, Lower = more blurry.
        
        Returns:
            Score 0-100
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        # Normalize to 0-100 (typical range for non-blurry: 50-100+)
        return float(min(100, variance / 100))
    
    @staticmethod
    def compute_contrast_score(frame: np.ndarray) -> float:
        """
        Compute contrast score using histogram spread.
        Higher = more contrast.
        
        Returns:
            Score 0-100
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        # Entropy as proxy for contrast
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return float(min(100, entropy / 8 * 100))
    
    @staticmethod
    def compute_lighting_score(frame: np.ndarray) -> float:
        """
        Score lighting consistency.
        Detects too bright/too dark/uneven lighting.
        
        Returns:
            Score 0-100
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Ideal brightness around 120-130
        ideal = 125
        deviation = abs(mean_brightness - ideal)
        
        score = 100 - (deviation / 255) * 100
        return float(max(0, min(100, score)))
    
    @staticmethod
    def analyze_video(
        video_path: str | Path,
        sample_frame_count: int = 30,
    ) -> dict:
        """
        Comprehensive video quality analysis.
        
        Returns:
            Dictionary with quality metrics
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample frames uniformly
        sample_indices = np.linspace(0, total_frames - 1, sample_frame_count, dtype=int)
        
        blur_scores = []
        contrast_scores = []
        lighting_scores = []
        
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            blur_scores.append(VideoQualityScorer.compute_blur_score(frame))
            contrast_scores.append(VideoQualityScorer.compute_contrast_score(frame))
            lighting_scores.append(VideoQualityScorer.compute_lighting_score(frame))
        
        cap.release()
        
        # Compute statistics
        return {
            'sharpness': {
                'mean': float(np.mean(blur_scores)),
                'min': float(np.min(blur_scores)),
                'max': float(np.max(blur_scores)),
                'std': float(np.std(blur_scores)),
                'status': 'good' if np.mean(blur_scores) > 50 else 'soft',
            },
            'contrast': {
                'mean': float(np.mean(contrast_scores)),
                'min': float(np.min(contrast_scores)),
                'max': float(np.max(contrast_scores)),
                'status': 'good' if np.mean(contrast_scores) > 40 else 'low',
            },
            'lighting': {
                'mean': float(np.mean(lighting_scores)),
                'min': float(np.min(lighting_scores)),
                'max': float(np.max(lighting_scores)),
                'std': float(np.std(lighting_scores)),
                'status': 'good' if np.std(lighting_scores) < 20 else 'uneven',
            },
            'fps': float(fps),
            'total_frames': total_frames,
            'duration_seconds': total_frames / fps if fps > 0 else 0,
        }


def estimate_sync_quality_from_segments(
    segments: list[dict],
    target_language: str,
) -> dict:
    """
    Estimate sync quality challenges based on segment characteristics.
    
    Returns:
        Dictionary with predicted challenges and recommendations
    """
    if not segments:
        return {}
    
    durations = [float(seg.get('end', 0)) - float(seg.get('start', 0)) for seg in segments]
    text_lengths = [len(str(seg.get('text', ''))) for seg in segments]
    
    challenges = []
    
    # Very short segments can be hard to sync
    short_segments = sum(1 for d in durations if d < 0.5)
    if short_segments > len(segments) * 0.3:
        challenges.append({
            'type': 'short_segments',
            'severity': 'medium',
            'description': f'{short_segments} very short segments detected',
            'recommendation': 'May require manual sync adjustment',
        })
    
    # Very long segments
    long_segments = sum(1 for d in durations if d > 5.0)
    if long_segments > 0:
        challenges.append({
            'type': 'long_segments',
            'severity': 'low',
            'description': f'{long_segments} segments > 5 seconds',
            'recommendation': 'Consider splitting manually for better sync',
        })
    
    # Rapid speech
    speech_rates = [l / d if d > 0 else 0 for l, d in zip(text_lengths, durations)]
    fast_speech = sum(1 for r in speech_rates if r > 20)  # chars per second
    if fast_speech > len(segments) * 0.2:
        challenges.append({
            'type': 'fast_speech',
            'severity': 'medium',
            'description': 'Fast speech detected in multiple segments',
            'recommendation': 'Use slower TTS speed for better lip-sync',
        })
    
    return {
        'estimated_challenges': challenges,
        'sync_difficulty': 'easy' if not challenges else ('medium' if len(challenges) < 2 else 'hard'),
        'total_segments': len(segments),
        'avg_segment_duration': float(np.mean(durations)),
        'avg_speech_rate': float(np.mean(speech_rates)),
    }
