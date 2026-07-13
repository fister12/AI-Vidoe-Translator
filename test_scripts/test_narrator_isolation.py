import json
import tempfile
from pathlib import Path

def test_exclude_interval_check():
    # Simple check matching inference.py frame exclusion logic
    exclude_intervals = [[1.5, 3.0], [5.0, 6.5]]
    fps = 25.0
    
    # We want to test timestamps
    # Frame 0 (t = 0.0s) -> False
    # Frame 40 (t = 1.6s) -> True
    # Frame 90 (t = 3.6s) -> False
    # Frame 130 (t = 5.2s) -> True
    
    def check_is_excluded(frame_idx, fps, intervals):
        t = frame_idx / fps
        for start, end in intervals:
            if start <= t <= end:
                return True
        return False

    assert not check_is_excluded(0, fps, exclude_intervals), "Frame 0 should not be excluded"
    assert check_is_excluded(40, fps, exclude_intervals), "Frame 40 should be excluded (t=1.6s)"
    assert not check_is_excluded(90, fps, exclude_intervals), "Frame 90 should not be excluded (t=3.6s)"
    assert check_is_excluded(130, fps, exclude_intervals), "Frame 130 should be excluded (t=5.2s)"
    print("[SUCCESS] Exclude interval mathematical check passed!")

def test_imports():
    try:
        from main import _auto_detect_narrators
        print("[SUCCESS] Import of _auto_detect_narrators succeeded!")
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        raise e

if __name__ == "__main__":
    test_exclude_interval_check()
    test_imports()
