"""
Frame-based visual pattern detection using OpenCV template matching.
Scans video frames for reference images and returns timestamps of matches.
Enhanced with multi-scale matching, feature-based matching, and robust preprocessing.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
import subprocess
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def _preprocess_image(img: np.ndarray, use_edges: bool = True) -> np.ndarray:
    """Apply preprocessing to improve matching robustness."""
    # CLAHE for contrast enhancement
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    if use_edges:
        gray = cv2.Canny(gray, 50, 150)

    return gray


def _pyramid_downsample(
    gray: np.ndarray, max_levels: int = 4, scale_factor: float = 0.75
) -> List[Tuple[float, np.ndarray]]:
    """Generate a multi-scale pyramid with finer granularity.
    Returns list of (scale_factor, scaled_image) pairs."""
    levels = [(1.0, gray)]
    for i in range(max_levels):
        prev = levels[-1][1]
        if prev.shape[0] < 32 or prev.shape[1] < 32:
            break
        new_scale = levels[-1][0] * scale_factor
        # Use resize for more precise scale control instead of pyrDown
        h = int(gray.shape[0] * new_scale)
        w = int(gray.shape[1] * new_scale)
        if h < 32 or w < 32:
            break
        resized = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
        levels.append((new_scale, resized))
    # Also add upscaling for when template is smaller than in-video element
    for i in range(2):
        prev = levels[0][1]
        new_scale = 1.0 + (i + 1) * 0.25
        h = int(gray.shape[0] * new_scale)
        w = int(gray.shape[1] * new_scale)
        resized = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)
        levels.append((new_scale, resized))
    return levels


def _feature_match(
    frame_gray: np.ndarray,
    template_gray: np.ndarray,
    min_matches: int = 10,
) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
    """
    Feature-based matching using ORB descriptors.
    Returns (confidence, bounding_box) or (0.0, None) if no match.
    """
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)

    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(frame_gray, None)

    if des1 is None or des2 is None or len(kp1) < min_matches or len(kp2) < min_matches:
        return 0.0, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except cv2.error:
        return 0.0, None

    # Lowe's ratio test
    good_matches = []
    for m_list in matches:
        if len(m_list) == 2:
            m, n = m_list
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    if len(good_matches) < min_matches:
        return 0.0, None

    # Compute bounding box from matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    try:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            return 0.0, None
    except cv2.error:
        return 0.0, None

    h, w = template_gray.shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(corners, M)

    x_min = max(0, int(dst[:,:,0].min()))
    y_min = max(0, int(dst[:,:,1].min()))
    x_max = int(dst[:,:,0].max())
    y_max = int(dst[:,:,1].max())

    # Confidence based on number of good matches relative to template features
    confidence = min(1.0, len(good_matches) / max(len(kp1), 20))

    return confidence, (x_min, y_min, x_max - x_min, y_max - y_min)


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


def _match_template_multi_scale(
    search_region: np.ndarray,
    template: np.ndarray,
    pyramid_levels: List[Tuple[float, np.ndarray]],
    use_preprocessing: bool = True,
) -> Tuple[float, Tuple[int, int]]:
    """
    Match template at multiple scales and preprocessing modes.
    Returns (best_confidence, best_location).
    """
    best_conf = 0.0
    best_loc = (0, 0)

    # Multiple preprocessing modes for robustness
    preprocess_modes = [False, True] if use_preprocessing else [False]

    for use_edges in preprocess_modes:
        proc_region = _preprocess_image(search_region, use_edges=use_edges) if use_edges else search_region
        proc_template = _preprocess_image(template, use_edges=use_edges) if use_edges else template

        for scale, scaled_tmpl in pyramid_levels:
            th, tw = scaled_tmpl.shape[:2]
            if th > proc_region.shape[0] or tw > proc_region.shape[1]:
                continue
            if th < 16 or tw < 16:
                continue

            scaled_tmpl_proc = cv2.resize(scaled_tmpl, (tw, th)) if use_edges else cv2.resize(scaled_tmpl, (tw, th))
            if len(proc_region.shape) == 3 and len(scaled_tmpl_proc.shape) == 2:
                continue
            if len(proc_region.shape) == 2 and len(scaled_tmpl_proc.shape) == 3:
                continue

            try:
                result = cv2.matchTemplate(proc_region, scaled_tmpl_proc, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val > best_conf:
                    best_conf = max_val
                    best_loc = max_loc
            except cv2.error:
                continue

    return best_conf, best_loc


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
                "threshold": float,      # Match confidence 0.0-1.0 (default 0.6)
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

    # Load template images with preprocessing
    templates = []
    for p in patterns:
        img_path = Path(p["image_path"])
        if not img_path.exists():
            logger.warning("Pattern image not found: %s", img_path)
            continue
        template_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if template_bgr is None:
            logger.warning("Could not read pattern image: %s", img_path)
            continue
        template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
        templates.append({
            "label": p.get("label", img_path.stem),
            "template_bgr": template_bgr,
            "template_gray": template_gray,
            "threshold": p.get("threshold", 0.6),
            "pre_seconds": p.get("pre_seconds", 60),
            "post_seconds": p.get("post_seconds", 60),
            "match_region": p.get("match_region"),
            "w": template_bgr.shape[1],
            "h": template_bgr.shape[0],
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

    # Increased processing resolution for better accuracy
    proc_width = 854
    proc_height = 480

    command = [
        "ffmpeg", "-v", "error", "-an", "-sn",
        "-i", str(video_path),
        "-vf", f"fps={fps / frame_sample_rate:.3f},scale={proc_width}:{proc_height}",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "-threads", "0",
        "-",
    ]

    logger.info(
        "Starting pattern detection: %d templates, %.1f fps (sample every %d frames), resolution %dx%d",
        len(templates), fps / frame_sample_rate, frame_sample_rate, proc_width, proc_height,
    )

    try:
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
    except Exception as exc:
        logger.error("Failed to start ffmpeg for pattern detection: %s", exc)
        return []

    frame_bytes = proc_width * proc_height * 3  # RGB24 = 3 bytes per pixel
    sample_step = max(1, int(fps * frame_sample_rate / fps))

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

            # Parse RGB frame
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(proc_height, proc_width, 3)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            for t_idx, tmpl in enumerate(templates):
                region = tmpl["match_region"]
                if region:
                    rx, ry, rw, rh = region
                    rx = int(rx * proc_width)
                    ry = int(ry * proc_height)
                    rw = int(rw * proc_width)
                    rh = int(rh * proc_height)
                    search_region = frame_gray[ry:ry + rh, rx:rx + rw]
                    search_region_color = frame[ry:ry + rh, rx:rx + rw]
                else:
                    search_region = frame_gray
                    search_region_color = frame
                    rx = ry = 0

                # Strategy 1: Multi-scale template matching with preprocessing
                pyramid = _pyramid_downsample(tmpl["template_gray"])
                best_conf, best_loc = _match_template_multi_scale(
                    search_region, tmpl["template_gray"], pyramid, use_preprocessing=True
                )

                # Strategy 2: Feature-based matching as fallback/supplement
                feat_conf, feat_bbox = _feature_match(search_region, tmpl["template_gray"], min_matches=8)

                # Use the better of the two methods
                if feat_conf > best_conf:
                    best_conf = feat_conf
                    if feat_bbox:
                        fx, fy, fw, fh = feat_bbox
                        best_loc = (fx, fy)

                # Strategy 3: Color histogram correlation (supplementary signal)
                if best_conf < tmpl["threshold"] and len(search_region_color.shape) == 3:
                    try:
                        hsv_search = cv2.cvtColor(search_region_color, cv2.COLOR_BGR2HSV)
                        hsv_template = cv2.cvtColor(tmpl["template_bgr"], cv2.COLOR_BGR2HSV)
                        h_hist = cv2.calcHist([hsv_search], [0], None, [50], [0, 180])
                        s_hist = cv2.calcHist([hsv_search], [1], None, [60], [0, 256])
                        h_hist_t = cv2.calcHist([hsv_template], [0], None, [50], [0, 180])
                        s_hist_t = cv2.calcHist([hsv_template], [1], None, [60], [0, 256])
                        cv2.normalize(h_hist, h_hist)
                        cv2.normalize(s_hist, s_hist)
                        cv2.normalize(h_hist_t, h_hist_t)
                        cv2.normalize(s_hist_t, s_hist_t)
                        color_score = (
                            cv2.compareHist(h_hist, h_hist_t, cv2.HISTCMP_CORREL) * 0.6 +
                            cv2.compareHist(s_hist, s_hist_t, cv2.HISTCMP_CORREL) * 0.4
                        )
                        if color_score > best_conf and color_score > 0.5:
                            best_conf = color_score
                            # For color match, use center of region as location
                            best_loc = (search_region.shape[1] // 4, search_region.shape[0] // 4)
                    except cv2.error:
                        pass

                if best_conf >= tmpl["threshold"]:
                    bx, by = best_loc
                    if region:
                        bx += rx
                        by += ry
                    cx = (bx + tmpl["w"] / 2) / proc_width
                    cy = (by + tmpl["h"] / 2) / proc_height
                    all_matches.append((best_conf, cx, cy, timestamp, t_idx))
                    logger.debug(
                        "Match at %.1fs: conf=%.3f pattern=%s",
                        timestamp, best_conf, tmpl["label"]
                    )

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
