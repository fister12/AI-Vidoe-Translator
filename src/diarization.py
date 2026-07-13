import numpy as np
import soundfile as sf
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

def diarize_and_extract_speakers(
    segments: list[dict],
    working_dir: Path,
    tts_model_name: str,
    device: str,
    num_speakers: int | None = None,
) -> int:
    """
    Perform speaker diarization on transcription segments.
    Extracts speaker embeddings using the XTTS model, clusters them,
    generates a combined voice sample for each speaker, and updates
    each segment's 'speaker_wav_path' to point to their corresponding
    speaker voice sample.
    
    Returns the number of detected speakers.
    """
    # Ensure Windows DLL search paths are loaded (crucial for torchcodec/torchaudio on Windows)
    try:
        from src.tts import _prepare_windows_dll_search_paths
        _prepare_windows_dll_search_paths()
    except Exception as e:
        print(f"  [WARN] Failed to prepare DLL search paths: {e}")

    # 1. Load XTTS model to extract embeddings
    from src.tts import _get_or_load_xtts_model
    gpu = (device == "cuda")
    tts = _get_or_load_xtts_model(tts_model_name, gpu=gpu)
    model = tts.synthesizer.tts_model
    
    # 2. Extract embeddings for each segment
    embeddings = []
    valid_indices = []
    
    print("\n[Diarization] Extracting speaker embeddings for segments...")
    for idx, seg in enumerate(segments):
        wav_path = seg.get("speaker_wav_path")
        if not wav_path or not Path(wav_path).exists():
            continue
            
        # Skip very short segments (e.g. less than 0.5s) for embedding calculation
        # to avoid noisy embeddings
        duration = float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
        if duration < 0.5:
            continue
            
        try:
            # We call get_conditioning_latents
            _, speaker_embedding = model.get_conditioning_latents(audio_path=[str(wav_path)])
            emb = speaker_embedding.cpu().squeeze().numpy()
            embeddings.append(emb)
            valid_indices.append(idx)
        except Exception as e:
            print(f"  [WARN] Failed to extract embedding for segment {idx}: {e}")
            
    if not embeddings:
        print("  [WARN] No valid speaker embeddings could be extracted.")
        return 1
        
    embeddings = np.array(embeddings)
    
    # 3. Cluster embeddings
    # Normalize to unit circle for cosine distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_embeddings = embeddings / norms
    
    n_samples = len(norm_embeddings)
    if n_samples < 2:
        num_clusters = 1
        labels = np.zeros(n_samples, dtype=int)
    elif num_speakers is not None and num_speakers > 0:
        num_clusters = min(num_speakers, n_samples)
        try:
            clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='cosine', linkage='average')
        except TypeError:
            clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='cosine', linkage='average')
        labels = clustering.fit_predict(norm_embeddings)
    else:
        # Auto-detect number of speakers using Silhouette Score
        best_k = 2
        best_score = -1.0
        max_k = min(6, n_samples)
        if max_k <= 2:
            num_clusters = 1
            labels = np.zeros(n_samples, dtype=int)
        else:
            for k_candidate in range(2, max_k):
                try:
                    clustering = AgglomerativeClustering(n_clusters=k_candidate, metric='cosine', linkage='average')
                except TypeError:
                    clustering = AgglomerativeClustering(n_clusters=k_candidate, affinity='cosine', linkage='average')
                l = clustering.fit_predict(norm_embeddings)
                score = silhouette_score(norm_embeddings, l, metric='cosine')
                if score > best_score:
                    best_score = score
                    best_k = k_candidate
            num_clusters = best_k
            try:
                clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='cosine', linkage='average')
            except TypeError:
                clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='cosine', linkage='average')
            labels = clustering.fit_predict(norm_embeddings)
            
    print(f"[Diarization] Detected {num_clusters} speaker(s).")
    
    # Map valid segment index to cluster label
    segment_labels = {}
    for idx, label in zip(valid_indices, labels):
        segment_labels[idx] = label
        
    # For any skipped segment (too short, etc.), assign to the closest valid segment's cluster in time
    for idx, seg in enumerate(segments):
        if idx not in segment_labels:
            if not valid_indices:
                segment_labels[idx] = 0
            else:
                closest_valid_idx = min(valid_indices, key=lambda x: abs(x - idx))
                segment_labels[idx] = segment_labels[closest_valid_idx]
                
    # 4. Generate combined voice sample for each speaker
    speaker_samples = {}
    speaker_dir = working_dir / "speaker_samples"
    speaker_dir.mkdir(parents=True, exist_ok=True)
    
    for c in range(num_clusters):
        cluster_seg_indices = [idx for idx, label in segment_labels.items() if label == c]
        # Sort by duration descending to get the best voice quality first
        cluster_segs_sorted = sorted(
            [(idx, segments[idx]) for idx in cluster_seg_indices],
            key=lambda item: float(item[1].get("end", 0.0)) - float(item[1].get("start", 0.0)),
            reverse=True
        )
        
        # Concatenate audio up to 12 seconds
        concatenated_wavs = []
        total_duration = 0.0
        
        for idx, seg in cluster_segs_sorted:
            wav_path = seg.get("speaker_wav_path")
            if wav_path and Path(wav_path).exists():
                duration = float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
                concatenated_wavs.append(wav_path)
                total_duration += duration
                if total_duration >= 12.0:
                    break
                    
        speaker_wav_output = speaker_dir / f"speaker_{c}.wav"
        
        if concatenated_wavs:
            # Concatenate WAV files
            waveforms = []
            sample_rate = None
            for p in concatenated_wavs:
                data, sr = sf.read(str(p))
                if sample_rate is None:
                    sample_rate = sr
                waveforms.append(data)
            
            concatenated = np.concatenate(waveforms, axis=0)
            sf.write(str(speaker_wav_output), concatenated, sample_rate)
            speaker_samples[c] = speaker_wav_output
            print(f"  [Diarization] Speaker {c} sample duration: {total_duration:.2f}s, saved to {speaker_wav_output.name}")
        else:
            # Fallback to global sample if no wav
            speaker_samples[c] = working_dir / "speaker_sample.wav"
            
    # 5. Update segment speaker_wav_path to point to speaker sample
    for idx, seg in enumerate(segments):
        label = int(segment_labels[idx])
        seg["speaker_wav_path"] = speaker_samples[label]
        seg["speaker_id"] = label
        
    return num_clusters
