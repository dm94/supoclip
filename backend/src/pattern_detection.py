"""
Frame-based visual pattern detection using OpenCV template matching.
Scans video frames for reference images and returns timestamps of matches.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
import subprocess
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def _pyramid_downsample(
    gray: np.ndarray, max_levels: int = 3
) -> List[Tuple[float, np.ndarray]]:
    """Generate a multi-scale pyramid for the image.
    Returns list of (scale_factor, scaled_image) pairs."""
    levels = [(1.0, gray)]
    for _ in range(max_levels):
        prev = levels[-1][1]
        if prev.shape[0] < 32 or prev.shape[1] < 32:
            break
        smaller = cv2.pyrDown(prev)
        levels.append((levels[-1][0] * 0.5, smaller))
    return levels


def _non_max_suppression(
    matches: List[Tuple[float, float, float, float, int]],
    min_distance: float = 0.5,
) -> List[Tuple[float, float, float, float, int]]:
    """Suppress nearby detections, keeping the highest-confidence one."""
    if not matches:
        return []
    sorted_matches = sorted(matches, key=lambda m: m[0], reverse=True)
    kept = []
    for conf, cx, cy, t, t_idx in sorted_matches:
        is_duplicate = False
        for _, kcx, kcy, kt, _ in kept:
            if abs(t - kt) < min_distance and abs(cx - kcx) < 50 and abs(cy - kcy) < 50:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append((conf, cx, cy, t, t_idx))
    return kept


def _seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS or HH:MM:SS format."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def detect_patterns_in_video(
    video_path: Path,
    patterns: List[Dict[str, Any]],
    frame_sample_rate: int = 5,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Scan video frames for visual patterns using OpenCV template matching.

    Args:
        video_path: Path to the video file.
        patterns: List of pattern config dicts:
            {
                "image_path": str,       # Path to reference image
                "label": str,            # Human-readable name
                "threshold": float,      # Match confidence 0.0-1.0 (default 0.8)
                "pre_seconds": int,      # Seconds before match for clip start
                "post_seconds": int,     # Seconds after match for clip end
                "match_region": list | None,  # [x, y, w, h] for ROI
            }
        frame_sample_rate: Check every N frames (higher = faster but less accurate).
        progress_callback: Optional fn(current_frame, total_frames).

    Returns:
        List of match dicts:
        {
            "timestamp": float,
            "timestamp_label": str,
            "confidence": float,
            "pattern_label": str,
            "bbox": [x, y, w, h],
            "pre_seconds": int,
            "post_seconds": int,
        }
    """
    if not patterns:
        logger.info("No patterns provided, skipping detection")
        return []

    # Load template images
    templates = []
    for p in patterns:
        img_path = Path(p["image_path"])
        if not img_path.exists():
            logger.warning("Pattern image not found: %s", img_path)
            continue
        template = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            logger.warning("Could not read pattern image: %s", img_path)
            continue
        templates.append({
            "label": p.get("label", img_path.stem),
            "template": template,
            "threshold": p.get("threshold", 0.8),
            "pre_seconds": p.get("pre_seconds", 60),
            "post_seconds": p.get("post_seconds", 60),
            "match_region": p.get("match_region"),
            "w": template.shape[1],
            "h": template.shape[0],
        })

    if not templates:
        logger.warning("No valid templates loaded")
        return []

    # Probe video to get FPS and frame count
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,nb_frames",
            "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    parts = probe.stdout.strip().split(",")
    fps = 30.0
    total_frames = 0
    if parts:
        try:
            num, den = parts[0].split("/")
            fps = float(num) / float(den) if float(den) > 0 else 30.0
        except (ValueError, IndexError, ZeroDivisionError):
            fps = 30.0
    if len(parts) > 1 and parts[1]:
        try:
            total_frames = int(parts[1])
        except ValueError:
            total_frames = 0

    if total_frames <= 0:
        duration_probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        try:
            duration = float(duration_probe.stdout.strip())
            total_frames = int(duration * fps)
        except (ValueError, IndexError):
            logger.error("Could not determine video duration")
            return []

    proc_width = 480
    proc_height = 270

    command = [
        "ffmpeg", "-v", "error", "-an", "-sn",
        "-i", str(video_path),
        "-vf", f"fps={fps / frame_sample_rate:.3f},scale={proc_width}:{proc_height}",
        "-pix_fmt", "gray",
        "-f", "rawvideo",
        "-threads", "0",
        "-",
    ]

    logger.info(
        "Starting pattern detection: %d templates, %.1f fps (sample every %d frames)",
        len(templates), fps / frame_sample_rate, frame_sample_rate,
    )

    try:
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
    except Exception as exc:
        logger.error("Failed to start ffmpeg for pattern detection: %s", exc)
        return []

    frame_bytes = proc_width * proc_height
    sample_step = int(fps * frame_sample_rate / fps)
    if sample_step < 1:
        sample_step = 1

    all_matches: List[Tuple[float, float, float, float, int]] = []
    frame_idx = 0
    sample_idx = 0

    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break

            timestamp = frame_idx / fps

            sample_idx += 1
            if sample_idx % sample_step != 0:
                frame_idx += 1
                continue

            if progress_callback and frame_idx % (fps * 10) < 1:
                progress_callback(frame_idx, total_frames)

            gray = np.frombuffer(raw, dtype=np.uint8).reshape(proc_height, proc_width)

            for t_idx, tmpl in enumerate(templates):
                region = tmpl["match_region"]
                if region:
                    rx, ry, rw, rh = region
                    rx = int(rx * proc_width)
                    ry = int(ry * proc_height)
                    rw = int(rw * proc_width)
                    rh = int(rh * proc_height)
                    search_region = gray[ry:ry + rh, rx:rx + rw]
                else:
                    search_region = gray
                    rx = ry = 0

                best_conf = 0.0
                best_loc = (0, 0)
                levels = _pyramid_downsample(tmpl["template"])

                for scale, scaled_tmpl in levels:
                    tw = int(scaled_tmpl.shape[1])
                    th = int(scaled_tmpl.shape[0])
                    if th > search_region.shape[0] or tw > search_region.shape[1]:
                        continue
                    result = cv2.matchTemplate(
                        search_region, scaled_tmpl, cv2.TM_CCOEFF_NORMED
                    )
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    if max_val > best_conf:
                        best_conf = max_val
                        best_loc = max_loc

                if best_conf >= tmpl["threshold"]:
                    bx, by = best_loc
                    if region:
                        bx += rx
                        by += ry
                    cx = (bx + tmpl["w"] / 2) / proc_width
                    cy = (by + tmpl["h"] / 2) / proc_height
                    all_matches.append((best_conf, cx, cy, timestamp, t_idx))

            frame_idx += 1

    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()

    logger.info("Pattern detection complete: %d raw matches", len(all_matches))

    if not all_matches:
        return []

    # Non-maximum suppression
    kept = _non_max_suppression(all_matches)

    results = []
    for conf, cx, cy, t, t_idx in kept:
        if t < 0:
            continue
        p = patterns[t_idx] if t_idx < len(patterns) else {}
        results.append({
            "timestamp": t,
            "timestamp_label": _seconds_to_timestamp(t),
            "confidence": round(float(conf), 3),
            "pattern_label": p.get("label", ""),
            "bbox": [float(cx), float(cy), 0.0, 0.0],
            "pre_seconds": p.get("pre_seconds", 60),
            "post_seconds": p.get("post_seconds", 60),
        })

    logger.info("Pattern detection: %d final matches after suppression", len(results))
    return results


def build_segments_from_matches(
    matches: List[Dict[str, Any]],
    min_gap_seconds: float = 5.0,
    max_duration: float = 300.0,
    min_duration: float = 10.0,
) -> List[Dict[str, Any]]:
    """
    Convert pattern matches into clip segments.
    Groups nearby matches and expands to [t - pre, t + post].

    Args:
        matches: List of match dicts from detect_patterns_in_video.
        min_gap_seconds: Merge matches closer than this.
        max_duration: Maximum clip duration in seconds.
        min_duration: Minimum clip duration in seconds.

    Returns:
        List of segment dicts compatible with the clip rendering pipeline:
        {
            "start_time": "MM:SS",
            "end_time": "MM:SS",
            "text": "Pattern: label at timestamp",
            "relevance_score": 1.0,
            "reasoning": "Visual pattern detection",
        }
    """
    if not matches:
        return []

    # Sort by timestamp
    sorted_matches = sorted(matches, key=lambda m: m["timestamp"])

    # Build time ranges
    ranges: List[Tuple[float, float, List[Dict]]] = []
    for match in sorted_matches:
        t = match["timestamp"]
        pre = match.get("pre_seconds", 60)
        post = match.get("post_seconds", 60)
        seg_start = max(0, t - pre)
        seg_end = t + post
        if seg_end - seg_start < min_duration:
            mid = (seg_start + seg_end) / 2
            seg_start = max(0, mid - min_duration / 2)
            seg_end = mid + min_duration / 2
        if seg_end - seg_start > max_duration:
            mid = (seg_start + seg_end) / 2
            seg_start = mid - max_duration / 2
            seg_end = mid + max_duration / 2

        if ranges and t - ranges[-1][1] < min_gap_seconds:
            prev_start, prev_end, prev_matches = ranges[-1]
            ranges[-1] = (min(prev_start, seg_start), max(prev_end, seg_end), prev_matches + [match])
        else:
            ranges.append((seg_start, seg_end, [match]))

    # Deduplicate overlapping ranges
    deduped: List[Tuple[float, float, List[Dict]]] = []
    for start, end, ms in ranges:
        if deduped and start < deduped[-1][1]:
            prev_start, prev_end, prev_ms = deduped[-1]
            deduped[-1] = (min(prev_start, start), max(prev_end, end), prev_ms + ms)
        else:
            deduped.append((start, end, ms))

    segments = []
    for i, (start, end, ms) in enumerate(deduped):
        labels = set(m["pattern_label"] or f"pattern at {m['timestamp_label']}" for m in ms)
        text = "Pattern detected: " + ", ".join(sorted(labels))
        segments.append({
            "clip_id": i + 1,
            "start_time": _seconds_to_timestamp(start),
            "end_time": _seconds_to_timestamp(end),
            "text": text,
            "relevance_score": 1.0,
            "reasoning": "Visual pattern detection",
            "virality_score": 0,
            "hook_score": 0,
            "engagement_score": 0,
            "value_score": 0,
            "shareability_score": 0,
            "hook_type": None,
        })

    logger.info("Built %d segments from %d pattern matches", len(segments), len(matches))
    return segments


def merge_segments(
    transcript_segments: List[Dict[str, Any]],
    pattern_segments: List[Dict[str, Any]],
    mode: str = "combined",
) -> List[Dict[str, Any]]:
    """
    Merge transcript-based segments with pattern-based segments.

    Args:
        transcript_segments: Segments from AI transcript analysis.
        pattern_segments: Segments from visual pattern detection.
        mode: "patterns_only" — only pattern segments
              "combined" — merge both, deduplicating overlaps
              "ai_only" — only transcript segments (passthrough)

    Returns:
        Merged list of segments.
    """
    if mode == "ai_only" or not pattern_segments:
        return transcript_segments
    if mode == "patterns_only":
        return pattern_segments

    combined = list(transcript_segments)

    def _parse_seconds(ts: str) -> float:
        parts = [int(p) for p in ts.split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        return 0.0

    existing_ranges = []
    for seg in combined:
        s = _parse_seconds(seg.get("start_time", "0:00"))
        e = _parse_seconds(seg.get("end_time", "0:00"))
        existing_ranges.append((s, e))

    for seg in pattern_segments:
        ps = _parse_seconds(seg.get("start_time", "0:00"))
        pe = _parse_seconds(seg.get("end_time", "0:00"))
        overlaps = any(ps < existing_end and pe > existing_start for existing_start, existing_end in existing_ranges)
        if not overlaps:
            combined.append(seg)

    combined.sort(key=lambda s: _parse_seconds(s.get("start_time", "0:00")))

    logger.info(
        "Merged segments: %d transcript + %d pattern = %d total (mode=%s)",
        len(transcript_segments), len(pattern_segments), len(combined), mode,
    )
    return combined
