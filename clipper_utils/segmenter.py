import cv2


def get_video_duration(video_path: str) -> float:
    """Returns total duration of a video file in seconds."""
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total / fps


def fixed_window_clips(duration_s: float,
                       clip_len_s: float = 4.0,
                       stride_s:   float = 2.0) -> list:
    """
    Splits a video of duration_s seconds into fixed-length overlapping clips.

    Args:
        duration_s : total video duration in seconds
        clip_len_s : length of each clip window (default 4s)
        stride_s   : stride between clip starts (default 2s → 50% overlap)

    Returns:
        List of (t_start, t_end) tuples in seconds.

    Example:
        duration=10s, clip_len=4s, stride=2s
        → [(0.0, 4.0), (2.0, 6.0), (4.0, 8.0), (6.0, 10.0), (8.0, 10.0)]
    """
    clips, t = [], 0.0
    while t + clip_len_s <= duration_s:
        clips.append((round(t, 2), round(t + clip_len_s, 2)))
        t += stride_s
    # Include trailing partial clip if >= 1 second remains
    if t < duration_s and (duration_s - t) >= 1.0:
        clips.append((round(t, 2), round(duration_s, 2)))
    return clips
