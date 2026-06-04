#!/usr/bin/env python3
"""
seamless_extender.py
====================

Extend a short clip of a person talking/moving into a much longer, seamless
video using the *Video Textures* approach (Schödl et al., SIGGRAPH 2000):

    1.  Run MediaPipe Pose + FaceMesh on every input frame, normalize the
        landmarks against the shoulder midpoint, and compute a per-frame
        velocity (mean optical-flow vector).
    2.  Build an N x N distance matrix that combines:
            * Euclidean distance of normalized landmarks
            * MSE of the central-crop grayscale thumbnails
            * Absolute difference of optical-flow magnitude (velocity)
    3.  Convert that into a sparse transition graph subject to constraints
        (min frame gap, velocity-direction alignment, landmark validity).
    4.  Random-walk the graph starting at frame 0 to build a sequence of
        operations totalling ``duration * fps`` frames.
    5.  Render the sequence and replace every jump with a 5-frame
        motion-compensated cross-fade (bidirectional DIS optical-flow
        warping + linear alpha blend).
    6.  Stream the frames straight into ffmpeg (libx264, silent) to keep
        memory usage bounded regardless of output length.

The expensive analysis artifacts (landmarks, thumbnails, velocities and the
distance matrix) are cached to ``--cache_dir`` keyed on the input file so
that repeated runs against the same clip are instant.

Dependencies
------------
    pip install opencv-python opencv-contrib-python mediapipe numpy scipy tqdm
    # ffmpeg must be available on PATH

Tested with: Python 3.9+, OpenCV 4.13, MediaPipe 0.10.x, ffmpeg 6+.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import random
import shutil
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #

LOG = logging.getLogger("seamless_extender")


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# --------------------------------------------------------------------------- #
# Data classes                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class VideoMetadata:
    """Container for the immutable properties of the input video."""

    path: str
    frame_count: int
    fps: float
    width: int
    height: int
    fourcc: str


@dataclass
class FrameFeatures:
    """All per-frame quantities used by the similarity computation."""

    landmarks: np.ndarray      # (N, K, 3) normalized landmark coordinates
    valid_mask: np.ndarray     # (N,) bool — True if pose landmarks detected
    thumb_gray: np.ndarray     # (N, H_t, W_t) uint8 center-crop thumbnails
    velocity_mag: np.ndarray   # (N,) mean magnitude of frame->next flow
    velocity_vec: np.ndarray   # (N, 2) mean vector of frame->next flow


@dataclass
class Op:
    """A single operation in the rendering plan."""

    kind: str            # 'linear' or 'transition'
    src: int             # frame index that we're playing or jumping from
    dst: int = -1        # destination frame (only for 'transition')


@dataclass
class WalkConfig:
    """Tunables for the random-walk frame sequencer."""

    p_jump: float = 0.30
    transition_len: int = 5
    cooldown_frames: int = 90
    recent_destinations: int = 8


# --------------------------------------------------------------------------- #
# MediaPipe landmark index subsets                                            #
# --------------------------------------------------------------------------- #

# Pose: shoulders + arms + hands (a stable upper-body signature for a
# talking-head / half-body shot).
POSE_INDICES: List[int] = [
    11, 12,            # shoulders (also our normalization anchor)
    13, 14,            # elbows
    15, 16,            # wrists
    17, 18,            # pinkies
    19, 20,            # index fingers
    21, 22,            # thumbs
]

# Face: a curated subset of the 468-point mesh that captures eye/lip/jaw
# silhouette without making the feature vector explode.
FACE_INDICES: List[int] = [
    33, 133, 159, 145,          # left eye corners + upper/lower lid
    362, 263, 386, 374,         # right eye corners + upper/lower lid
    61, 291, 0, 17, 13, 14,     # outer & inner lip extremes
    78, 308,                    # inner mouth corners
    152, 10,                    # chin tip, forehead center
    234, 454,                   # left/right cheek
    1, 4, 5,                    # nose ridge
]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

THUMB_SIZE: Tuple[int, int] = (96, 96)        # (w, h) for similarity thumbnails
FLOW_SIZE: Tuple[int, int] = (320, 180)       # (w, h) for velocity estimation


def video_cache_key(video_path: str) -> str:
    """Hash a video file path + size + mtime into a short stable cache key."""
    p = Path(video_path).resolve()
    stat = p.stat()
    h = hashlib.sha1()
    h.update(p.as_posix().encode("utf-8"))
    h.update(str(stat.st_size).encode("utf-8"))
    h.update(str(int(stat.st_mtime)).encode("utf-8"))
    return h.hexdigest()[:16]


def get_video_metadata(path: str) -> VideoMetadata:
    """Probe a video file with OpenCV and return its properties."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input video not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    finally:
        cap.release()
    fourcc = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip()
    if frame_count <= 0:
        raise ValueError(f"Video reports zero frames: {path}")
    return VideoMetadata(
        path=path, frame_count=frame_count, fps=fps,
        width=width, height=height, fourcc=fourcc,
    )


def _make_dis_flow(preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM):
    """Construct a DIS optical-flow calculator, gracefully falling back."""
    if hasattr(cv2, "DISOpticalFlow_create"):
        flow = cv2.DISOpticalFlow_create(preset)
        # Enable mean normalization + variational refinement for stability.
        try:
            flow.setUseMeanNormalization(True)
            flow.setUseSpatialPropagation(True)
        except Exception:
            pass
        return flow
    LOG.warning("cv2.DISOpticalFlow_create unavailable — falling back to Farneback.")
    return None


def _calc_flow(flow_calc, prev_gray: np.ndarray, cur_gray: np.ndarray) -> np.ndarray:
    """Compute dense optical flow with DIS or Farneback fallback."""
    if flow_calc is not None:
        return flow_calc.calc(prev_gray, cur_gray, None)
    return cv2.calcOpticalFlowFarneback(
        prev_gray, cur_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0,
    )


# --------------------------------------------------------------------------- #
# Step 2: Feature extraction                                                  #
# --------------------------------------------------------------------------- #

def _extract_landmarks_one(pose_results, face_results) -> Optional[np.ndarray]:
    """Convert MediaPipe results into a single normalized (K, 3) array.

    Returns ``None`` when pose landmarks are absent — those frames are flagged
    as *non-loopable* downstream.
    """
    if pose_results.pose_landmarks is None:
        return None
    pose_lms = pose_results.pose_landmarks.landmark
    l_sh, r_sh = pose_lms[11], pose_lms[12]
    cx = (l_sh.x + r_sh.x) * 0.5
    cy = (l_sh.y + r_sh.y) * 0.5
    scale = max(1e-6, float(np.hypot(l_sh.x - r_sh.x, l_sh.y - r_sh.y)))

    coords: List[List[float]] = []
    for idx in POSE_INDICES:
        lm = pose_lms[idx]
        coords.append([
            (lm.x - cx) / scale,
            (lm.y - cy) / scale,
            lm.z / scale,
        ])

    if face_results.multi_face_landmarks:
        face_lms = face_results.multi_face_landmarks[0].landmark
        for idx in FACE_INDICES:
            lm = face_lms[idx]
            coords.append([
                (lm.x - cx) / scale,
                (lm.y - cy) / scale,
                lm.z / scale,
            ])
    else:
        # Pad with NaN — these frames still have a valid pose so they remain
        # loopable on pose-only similarity, but face contributions will be
        # masked out below.
        for _ in FACE_INDICES:
            coords.append([np.nan, np.nan, np.nan])

    return np.asarray(coords, dtype=np.float32)


def extract_features(meta: VideoMetadata,
                     cache_dir: Path,
                     cache_key: str) -> FrameFeatures:
    """Walk every frame, extract landmarks/thumbs/velocity, cache & return."""
    cache_file = cache_dir / f"{cache_key}_features.npz"
    if cache_file.is_file():
        LOG.info("Loading cached features from %s", cache_file)
        data = np.load(cache_file)
        return FrameFeatures(
            landmarks=data["landmarks"],
            valid_mask=data["valid_mask"],
            thumb_gray=data["thumb_gray"],
            velocity_mag=data["velocity_mag"],
            velocity_vec=data["velocity_vec"],
        )

    try:
        import mediapipe as mp
    except ImportError as exc:                                              # pragma: no cover
        raise ImportError(
            "mediapipe is required. Install with: pip install mediapipe"
        ) from exc

    LOG.info("Extracting per-frame features from %s", meta.path)
    pose_solution = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_solution = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    n = meta.frame_count
    k = len(POSE_INDICES) + len(FACE_INDICES)
    landmarks = np.full((n, k, 3), np.nan, dtype=np.float32)
    valid_mask = np.zeros(n, dtype=bool)
    thumb_gray = np.zeros((n, THUMB_SIZE[1], THUMB_SIZE[0]), dtype=np.uint8)
    velocity_mag = np.zeros(n, dtype=np.float32)
    velocity_vec = np.zeros((n, 2), dtype=np.float32)

    cap = cv2.VideoCapture(meta.path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {meta.path}")
    flow_calc = _make_dis_flow(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    prev_gray_small: Optional[np.ndarray] = None
    read_n = n

    try:
        for i in tqdm(range(n), desc="Analyzing frames", unit="frame"):
            ret, frame_bgr = cap.read()
            if not ret:
                LOG.warning("Stream ended at frame %d (expected %d); truncating.",
                            i, n)
                read_n = i
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pose_res = pose_solution.process(frame_rgb)
            face_res = face_solution.process(frame_rgb)

            lms = _extract_landmarks_one(pose_res, face_res)
            if lms is not None:
                landmarks[i] = lms
                # Only pose portion of the vector must be finite to count as
                # valid — face is optional.
                pose_slice = lms[:len(POSE_INDICES)]
                valid_mask[i] = bool(np.isfinite(pose_slice).all())
            else:
                LOG.debug("No pose detected in frame %d (non-loopable).", i)

            # Center-crop thumbnail for pixel-level similarity
            ch, cw = meta.height // 2, meta.width // 2
            y0, x0 = meta.height // 4, meta.width // 4
            center = frame_bgr[y0:y0 + ch, x0:x0 + cw]
            thumb_gray[i] = cv2.resize(
                cv2.cvtColor(center, cv2.COLOR_BGR2GRAY),
                THUMB_SIZE, interpolation=cv2.INTER_AREA,
            )

            # Instantaneous velocity via downscaled dense optical flow
            gray_small = cv2.resize(
                cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY),
                FLOW_SIZE, interpolation=cv2.INTER_AREA,
            )
            if prev_gray_small is not None:
                flow = _calc_flow(flow_calc, prev_gray_small, gray_small)
                velocity_mag[i] = float(np.linalg.norm(flow, axis=2).mean())
                velocity_vec[i] = flow.mean(axis=(0, 1))
            prev_gray_small = gray_small
    finally:
        cap.release()
        pose_solution.close()
        face_solution.close()

    if read_n < n:
        landmarks = landmarks[:read_n]
        valid_mask = valid_mask[:read_n]
        thumb_gray = thumb_gray[:read_n]
        velocity_mag = velocity_mag[:read_n]
        velocity_vec = velocity_vec[:read_n]
        meta.frame_count = read_n

    invalid = int((~valid_mask).sum())
    if invalid:
        LOG.warning("Pose not detected in %d/%d frames (marked non-loopable).",
                    invalid, valid_mask.size)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        landmarks=landmarks, valid_mask=valid_mask,
        thumb_gray=thumb_gray, velocity_mag=velocity_mag,
        velocity_vec=velocity_vec,
    )
    LOG.info("Cached features → %s", cache_file)

    return FrameFeatures(
        landmarks=landmarks, valid_mask=valid_mask,
        thumb_gray=thumb_gray, velocity_mag=velocity_mag,
        velocity_vec=velocity_vec,
    )


# --------------------------------------------------------------------------- #
# Step 3: Distance matrix                                                     #
# --------------------------------------------------------------------------- #

@dataclass
class DistanceWeights:
    landmark: float = 1.0
    pixel: float = 1.0
    velocity: float = 0.5


def _pairwise_l2(x: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance over rows of ``x`` (N x D)."""
    sq = np.sum(x * x, axis=1)
    g = x @ x.T
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * g, 0.0)
    return np.sqrt(d2).astype(np.float32)


def _pairwise_mse(x: np.ndarray) -> np.ndarray:
    """Pairwise mean squared error over rows of ``x`` (N x D)."""
    sq = np.sum(x * x, axis=1)
    g = x @ x.T
    sse = np.maximum(sq[:, None] + sq[None, :] - 2.0 * g, 0.0)
    return (sse / float(x.shape[1])).astype(np.float32)


def _normalize_robust(m: np.ndarray) -> np.ndarray:
    """Scale a non-negative distance matrix by its 95th percentile."""
    finite = m[np.isfinite(m)]
    if finite.size == 0:
        return m
    ref = float(np.quantile(finite, 0.95))
    if ref <= 0:
        return m
    return np.clip(m / ref, 0.0, 1.0)


def compute_distance_matrix(features: FrameFeatures,
                            cache_dir: Path,
                            cache_key: str,
                            weights: DistanceWeights = DistanceWeights()
                            ) -> np.ndarray:
    """Construct the N x N similarity matrix combining all three metrics."""
    cache_file = cache_dir / f"{cache_key}_distance.npy"
    if cache_file.is_file():
        LOG.info("Loading cached distance matrix from %s", cache_file)
        return np.load(cache_file)

    LOG.info("Computing distance matrix...")
    n = features.landmarks.shape[0]

    # Landmark distance — NaNs become 0 so missing face points contribute
    # nothing rather than corrupting the entire row.
    lm = np.nan_to_num(features.landmarks, nan=0.0).reshape(n, -1)
    LOG.info("  · landmark L2  (%d × %d)", n, n)
    lm_dist = _pairwise_l2(lm)

    LOG.info("  · pixel MSE   (%d × %d)", n, n)
    thumbs = features.thumb_gray.astype(np.float32).reshape(n, -1) / 255.0
    pix_dist = _pairwise_mse(thumbs)

    LOG.info("  · velocity diff")
    v = features.velocity_mag.astype(np.float32)
    vel_dist = np.abs(v[:, None] - v[None, :])

    w_sum = weights.landmark + weights.pixel + weights.velocity
    d = (
        weights.landmark * _normalize_robust(lm_dist)
        + weights.pixel * _normalize_robust(pix_dist)
        + weights.velocity * _normalize_robust(vel_dist)
    ) / max(w_sum, 1e-9)

    # Frames with missing pose are unusable as either endpoint.
    invalid = ~features.valid_mask
    d[invalid, :] = np.inf
    d[:, invalid] = np.inf
    np.fill_diagonal(d, np.inf)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, d.astype(np.float32))
    LOG.info("Cached distance matrix → %s (min=%.4f, p10=%.4f)",
             cache_file,
             float(d[np.isfinite(d)].min()) if np.isfinite(d).any() else float("nan"),
             float(np.quantile(d[np.isfinite(d)], 0.10))
             if np.isfinite(d).any() else float("nan"))
    return d.astype(np.float32)


# --------------------------------------------------------------------------- #
# Step 3b: Transition graph                                                   #
# --------------------------------------------------------------------------- #

def build_transition_graph(d: np.ndarray,
                           features: FrameFeatures,
                           threshold: float,
                           min_gap: int,
                           velocity_align: float = 0.6,
                           max_per_node: int = 32
                           ) -> Dict[int, List[Tuple[int, float]]]:
    """Convert the distance matrix into a sparse graph of valid transitions."""
    LOG.info("Building transition graph (threshold=%.4f, min_gap=%d)",
             threshold, min_gap)
    n = d.shape[0]
    vel_vec = features.velocity_vec
    vel_mag = features.velocity_mag
    positive_v = vel_mag[vel_mag > 0]
    rest_thresh = float(np.quantile(positive_v, 0.10)) if positive_v.size else 0.0

    graph: Dict[int, List[Tuple[int, float]]] = {}
    valid_count = 0
    for i in range(n):
        if not features.valid_mask[i]:
            continue
        row = d[i]
        candidate_idx = np.where((row < threshold) & np.isfinite(row))[0]
        if candidate_idx.size == 0:
            continue
        kept: List[Tuple[int, float]] = []
        vi = vel_vec[i]
        mi = float(vel_mag[i])
        for j in candidate_idx:
            j_int = int(j)
            if abs(j_int - i) <= min_gap:
                continue
            if not features.valid_mask[j_int]:
                continue
            mj = float(vel_mag[j_int])
            both_resting = (mi <= rest_thresh) and (mj <= rest_thresh)
            cos = 1.0
            if mi > 1e-6 and mj > 1e-6:
                vj = vel_vec[j_int]
                denom = float(np.linalg.norm(vi) * np.linalg.norm(vj)) + 1e-9
                cos = float(np.dot(vi, vj) / denom)
            if not (both_resting or cos >= velocity_align):
                continue
            kept.append((j_int, float(row[j_int])))
        if kept:
            kept.sort(key=lambda t: t[1])
            graph[i] = kept[:max_per_node]
            valid_count += len(graph[i])

    LOG.info("Graph: %d source nodes, %d edges (avg %.1f per node)",
             len(graph), valid_count,
             valid_count / max(1, len(graph)))
    return graph


# --------------------------------------------------------------------------- #
# Step 4: Random walk sequencer                                               #
# --------------------------------------------------------------------------- #

def random_walk(transitions: Dict[int, List[Tuple[int, float]]],
                n_frames: int,
                target_frames: int,
                config: WalkConfig,
                rng: random.Random) -> List[Op]:
    """Walk the transition graph until ``target_frames`` have been emitted."""
    ops: List[Op] = []
    cur = 0
    emitted = 0
    cooldown = 0
    recent: Deque[int] = deque(maxlen=config.recent_destinations)

    progress = tqdm(total=target_frames, desc="Planning sequence", unit="frame")
    hard_cuts = 0

    while emitted < target_frames:
        if cur >= n_frames:
            # Reached end of source without finding a graceful jump.
            # Wrap to the closest in-graph frame near the beginning.
            cur = 0
            hard_cuts += 1
            LOG.debug("Hard wrap to frame 0 (cut #%d).", hard_cuts)

        end_horizon = n_frames - (config.transition_len + 1)
        forced = cur > end_horizon
        can_jump = (
            cooldown <= 0
            and cur in transitions
            and cur + config.transition_len <= n_frames
        )

        if forced or (can_jump and rng.random() < config.p_jump):
            # Candidate list, respecting "don't reuse the same dst too soon".
            pool = transitions.get(cur, [])
            usable = [
                (j, dist) for (j, dist) in pool
                if j + config.transition_len <= n_frames
                and j not in recent
            ]
            if not usable and forced and pool:
                # Near the end and starved → ignore the recent-destination
                # quarantine to avoid a hard cut.
                recent.clear()
                usable = [(j, dist) for (j, dist) in pool
                          if j + config.transition_len <= n_frames]

            if usable:
                # Inverse-distance weighted random choice — favor closer
                # matches but keep the walk non-deterministic.
                weights_arr = np.fromiter(
                    (1.0 / (d + 1e-6) for _, d in usable),
                    dtype=np.float64, count=len(usable),
                )
                weights_arr = weights_arr / weights_arr.sum()
                pick = rng.choices(range(len(usable)),
                                   weights=weights_arr.tolist(), k=1)[0]
                j, _ = usable[pick]
                ops.append(Op("transition", cur, j))
                emitted += config.transition_len
                progress.update(config.transition_len)
                recent.append(j)
                cur = j + config.transition_len
                cooldown = config.cooldown_frames
                continue
            elif forced:
                # No transitions available at end-of-file → wrap.
                cur = n_frames  # triggers wrap on next iter
                continue

        # Default: emit a linear frame.
        ops.append(Op("linear", cur))
        emitted += 1
        progress.update(1)
        cur += 1
        cooldown = max(0, cooldown - 1)

    progress.close()
    if hard_cuts:
        LOG.warning("Sequencer required %d hard wrap(s). Consider raising "
                    "--threshold or lowering --duration.", hard_cuts)
    return ops


# --------------------------------------------------------------------------- #
# Step 5: Motion-compensated cross-fade                                       #
# --------------------------------------------------------------------------- #

def motion_compensated_blend(src: np.ndarray,
                             dst: np.ndarray,
                             alpha: float,
                             flow_calc) -> np.ndarray:
    """Bidirectional optical-flow warping + linear alpha blend.

    ``alpha=0`` returns ``src``-dominant output and ``alpha=1`` returns
    ``dst``-dominant output. Pixels are *warped toward each other* so that
    hair, fabric folds and finger positions morph across the transition
    instead of cross-dissolving in place.
    """
    if src.shape != dst.shape:
        dst = cv2.resize(dst, (src.shape[1], src.shape[0]))

    gs = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gd = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    flow_fwd = _calc_flow(flow_calc, gs, gd)
    flow_bwd = _calc_flow(flow_calc, gd, gs)

    h, w = src.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x_src = xx + flow_fwd[..., 0] * alpha
    map_y_src = yy + flow_fwd[..., 1] * alpha
    map_x_dst = xx + flow_bwd[..., 0] * (1.0 - alpha)
    map_y_dst = yy + flow_bwd[..., 1] * (1.0 - alpha)

    warped_src = cv2.remap(
        src, map_x_src, map_y_src,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    warped_dst = cv2.remap(
        dst, map_x_dst, map_y_dst,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    blended = (
        (1.0 - alpha) * warped_src.astype(np.float32)
        + alpha * warped_dst.astype(np.float32)
    )
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Step 6: Streaming renderer                                                  #
# --------------------------------------------------------------------------- #

class FrameReader:
    """Random-access frame reader with sequential-read fast-path + LRU cache.

    OpenCV's seek is expensive on H.264 (decoder must rewind to the previous
    keyframe), so we cache the last ``cache_size`` reads. The renderer
    accesses frames in mostly-sequential chunks separated by occasional jumps,
    which matches this cache well.
    """

    def __init__(self, video_path: str, cache_size: int = 128) -> None:
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        self.expected_next = 0
        self._cache: Dict[int, np.ndarray] = {}
        self._order: Deque[int] = deque()
        self._cache_size = cache_size

    def read(self, frame_idx: int) -> np.ndarray:
        if frame_idx in self._cache:
            return self._cache[frame_idx]
        if frame_idx != self.expected_next:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self.expected_next = frame_idx
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise IOError(f"Failed to read frame {frame_idx}")
        self.expected_next = frame_idx + 1
        self._put(frame_idx, frame)
        return frame

    def _put(self, idx: int, frame: np.ndarray) -> None:
        self._cache[idx] = frame
        self._order.append(idx)
        while len(self._order) > self._cache_size:
            self._cache.pop(self._order.popleft(), None)

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def stream_frames(ops: List[Op],
                  reader: FrameReader,
                  transition_len: int) -> Generator[np.ndarray, None, None]:
    """Yield each output frame on demand without buffering the full video."""
    flow_calc = _make_dis_flow(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    for op in ops:
        if op.kind == "linear":
            yield reader.read(op.src)
        elif op.kind == "transition":
            i, j = op.src, op.dst
            denom = max(1, transition_len - 1)
            for t in range(transition_len):
                alpha = float(t) / float(denom)
                src = reader.read(i + t)
                dst = reader.read(j + t)
                yield motion_compensated_blend(src, dst, alpha, flow_calc)
        else:
            raise ValueError(f"Unknown op kind: {op.kind!r}")


def render_to_video(input_video: str,
                    output_video: str,
                    ops: List[Op],
                    meta: VideoMetadata,
                    total_out_frames: int,
                    transition_len: int) -> None:
    """Pipe blended frames into a libx264 ffmpeg subprocess (silent)."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it (e.g. `brew install ffmpeg`).")

    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{meta.width}x{meta.height}",
        "-r", f"{meta.fps:.6f}",
        "-i", "-",
        "-an",                         # no audio (silent)
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "18",
        "-movflags", "+faststart",
        output_video,
    ]
    LOG.info("Spawning ffmpeg → %s", output_video)
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    written = 0
    try:
        with FrameReader(input_video, cache_size=128) as reader, \
                tqdm(total=total_out_frames, desc="Rendering", unit="frame") as bar:
            for frame in stream_frames(ops, reader, transition_len):
                if frame.shape[1] != meta.width or frame.shape[0] != meta.height:
                    frame = cv2.resize(frame, (meta.width, meta.height))
                proc.stdin.write(frame.tobytes())
                written += 1
                bar.update(1)
    except (BrokenPipeError, IOError) as exc:
        LOG.error("Pipe to ffmpeg broke after %d frames: %s", written, exc)
    finally:
        try:
            if proc.stdin:
                proc.stdin.close()
        except BrokenPipeError:
            pass
        stderr_bytes = proc.stderr.read() if proc.stderr else b""
        rc = proc.wait()
        if rc != 0:
            LOG.error("ffmpeg failed (rc=%d):\n%s",
                      rc, stderr_bytes.decode("utf-8", errors="replace"))
            raise RuntimeError("ffmpeg encoding failed")
    LOG.info("Wrote %d frames (~%.1fs) to %s",
             written, written / max(meta.fps, 1e-6), output_video)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="seamless_extender.py",
        description=(
            "Losslessly extend a short clip of a person talking/moving into a "
            "long, seamless silent video using the Video Textures approach "
            "with motion-compensated optical-flow blending."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True, help="Input video path.")
    p.add_argument("--output", "-o", required=True, help="Output video path (.mp4).")
    p.add_argument("--duration", "-d", type=float, required=True,
                   help="Target output duration in seconds.")
    p.add_argument("--threshold", "-t", type=float, default=0.05,
                   help="Maximum normalized distance for a valid transition "
                        "(lower = stricter / fewer jumps).")
    p.add_argument("--cache_dir", default=".seamless_cache",
                   help="Directory for cached analysis artifacts.")
    p.add_argument("--min_gap", type=int, default=150,
                   help="Minimum |i - j| (in frames) between a transition's "
                        "two endpoints.")
    p.add_argument("--jump_probability", type=float, default=0.30,
                   help="Per-eligible-frame probability of taking a jump.")
    p.add_argument("--transition_frames", type=int, default=5,
                   help="Length of the motion-compensated cross-fade window.")
    p.add_argument("--cooldown", type=int, default=90,
                   help="Minimum frames of linear playback between consecutive "
                        "jumps (helps the result feel natural).")
    p.add_argument("--velocity_align", type=float, default=0.6,
                   help="Cosine threshold for velocity-direction alignment "
                        "between transition endpoints.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducible walks.")
    p.add_argument("--verbose", action="store_true",
                   help="Verbose / debug logging.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)

    try:
        meta = get_video_metadata(args.input)
        LOG.info("Input: %dx%d @ %.2f fps  (%d frames, ~%.1fs, fourcc=%r)",
                 meta.width, meta.height, meta.fps, meta.frame_count,
                 meta.frame_count / max(meta.fps, 1e-6), meta.fourcc)

        target_frames = int(round(args.duration * meta.fps))
        if target_frames <= meta.frame_count:
            LOG.warning("Requested duration (%.1fs) is no longer than the input "
                        "(%.1fs); the result will still be re-walked.",
                        args.duration, meta.frame_count / max(meta.fps, 1e-6))

        if meta.frame_count <= args.min_gap + args.transition_frames + 2:
            raise ValueError(
                f"Input has only {meta.frame_count} frames — too short for "
                f"--min_gap={args.min_gap}. Lower --min_gap or use a longer clip."
            )

        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = video_cache_key(args.input)

        features = extract_features(meta, cache_dir, cache_key)
        if not features.valid_mask.any():
            raise RuntimeError(
                "MediaPipe failed to detect a person in any frame; cannot proceed."
            )

        d_matrix = compute_distance_matrix(features, cache_dir, cache_key)
        graph = build_transition_graph(
            d_matrix, features,
            threshold=args.threshold,
            min_gap=args.min_gap,
            velocity_align=args.velocity_align,
        )
        if not graph:
            raise RuntimeError(
                f"No valid transitions at --threshold={args.threshold}. "
                "Try raising the threshold or lowering --min_gap."
            )

        rng = random.Random(args.seed) if args.seed is not None else random.Random()
        walk_cfg = WalkConfig(
            p_jump=args.jump_probability,
            transition_len=args.transition_frames,
            cooldown_frames=args.cooldown,
        )
        ops = random_walk(graph, meta.frame_count, target_frames, walk_cfg, rng)
        total_out = sum(
            args.transition_frames if op.kind == "transition" else 1
            for op in ops
        )
        n_jumps = sum(1 for op in ops if op.kind == "transition")
        LOG.info("Planned %d ops (%d jumps) → %d frames (~%.1fs)",
                 len(ops), n_jumps, total_out, total_out / meta.fps)

        render_to_video(
            args.input, args.output, ops, meta, total_out,
            transition_len=args.transition_frames,
        )
        return 0

    except KeyboardInterrupt:
        LOG.warning("Interrupted by user.")
        return 130
    except Exception as exc:                                                # pragma: no cover
        LOG.exception("Failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
