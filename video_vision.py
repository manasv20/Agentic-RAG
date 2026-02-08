"""
Video visual vectorization: extract frames from video and embed with CLIP for semantic search.
Requires: opencv-python-headless, sentence-transformers (and torch).
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from typing import List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

# Lazy imports to avoid loading heavy deps until needed
_cv2 = None
_clip_model = None


def _get_cv2():
    global _cv2
    if _cv2 is None:
        try:
            import cv2
            _cv2 = cv2
        except ImportError:
            raise ImportError("opencv-python-headless is required for video frame extraction. pip install opencv-python-headless")
    return _cv2


def _get_clip():
    global _clip_model
    if _clip_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _clip_model = SentenceTransformer("clip-ViT-B-32")
        except Exception as e:
            raise ImportError("sentence-transformers is required for video embeddings. pip install sentence-transformers") from e
    return _clip_model


def get_video_duration(video_path: str) -> float:
    """Return video duration in seconds. Raises if video cannot be opened."""
    # #region agent log
    try:
        _lp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cursor", "debug.log")
        open(_lp, "a").write(__import__("json").dumps({"hypothesisId": "H1", "location": "video_vision.py:get_video_duration_entry", "message": "get_video_duration entry", "data": {"video_path": video_path, "exists": os.path.exists(video_path)}, "timestamp": int(__import__("time").time() * 1000), "runId": "video_workflow"}) + "\n")
    except Exception:
        pass
    # #endregion
    cv2 = _get_cv2()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # #region agent log
        try:
            _lp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cursor", "debug.log")
            open(_lp, "a").write(__import__("json").dumps({"hypothesisId": "H1", "location": "video_vision.py:get_video_duration_not_opened", "message": "VideoCapture not opened", "data": {"video_path": video_path}, "timestamp": int(__import__("time").time() * 1000), "runId": "video_workflow"}) + "\n")
        except Exception:
            pass
        # #endregion
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if video_fps <= 0:
            return 0.0
        out = frame_count / video_fps
        # #region agent log
        try:
            _lp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cursor", "debug.log")
            open(_lp, "a").write(__import__("json").dumps({"hypothesisId": "H1", "location": "video_vision.py:get_video_duration_ok", "message": "get_video_duration success", "data": {"duration": out}, "timestamp": int(__import__("time").time() * 1000), "runId": "video_workflow"}) + "\n")
        except Exception:
            pass
        # #endregion
        return out
    finally:
        cap.release()


def extract_frames(video_path: str, fps: float = 1.0, max_frames: int = 100) -> List[Tuple[float, bytes]]:
    """
    Extract frames from a video file at roughly the given FPS.
    Returns list of (timestamp_sec, jpeg_bytes) for each frame.
    """
    cv2 = _get_cv2()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        interval = max(1, int(video_fps / fps))
        result = []
        frame_idx = 0
        saved = 0
        while saved < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                ts = frame_idx / video_fps
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, jpeg = cv2.imencode(".jpg", frame_rgb)
                result.append((ts, jpeg.tobytes()))
                saved += 1
            frame_idx += 1
        return result
    finally:
        cap.release()


def embed_frames(frame_jpeg_bytes_list: List[bytes]) -> List[List[float]]:
    """Encode video frames (as JPEG bytes) to CLIP embeddings."""
    import io
    from PIL import Image
    model = _get_clip()
    images = [Image.open(io.BytesIO(b)).convert("RGB") for b in frame_jpeg_bytes_list]
    embeddings = model.encode(images, convert_to_numpy=True)
    return embeddings.tolist()


def get_clip_text_embedding_function():
    """Return a ChromaDB-compatible embedding function that encodes text with CLIP (for querying)."""
    class ClipTextEmbeddingFunction:
        def __init__(self):
            self._model = None

        def _model_load(self):
            if self._model is None:
                self._model = _get_clip()

        def __call__(self, input: List[str]) -> List[List[float]]:
            self._model_load()
            emb = self._model.encode(input, convert_to_numpy=True)
            return emb.tolist()

    return ClipTextEmbeddingFunction()


def add_video_frames_to_chroma(
    video_path: str,
    collection_name: str,
    fps: float = 1.0,
    max_frames: int = 100,
    progress_callback=None,
    CHROMA_PATH: str = "./chroma_db",
) -> "Any":
    """
    Extract frames from video, embed with CLIP, store in ChromaDB.
    Collection uses CLIP text embedding for queries (same space as image embeddings).
    Returns the ChromaDB collection.
    """
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        raise ImportError("chromadb is required")

    frames_with_ts = extract_frames(video_path, fps=fps, max_frames=max_frames)
    if not frames_with_ts:
        raise RuntimeError("No frames extracted from video")

    frame_bytes = [b for _, b in frames_with_ts]
    timestamps = [t for t, _ in frames_with_ts]
    total = len(frame_bytes)

    if progress_callback:
        progress_callback(0, total)
    embeddings = embed_frames(frame_bytes)
    if progress_callback:
        progress_callback(total, total)

    ids = [f"frame_{i}_{hashlib.sha256(frame_bytes[i]).hexdigest()[:12]}" for i in range(total)]
    metadatas = [{"frame_index": i, "time_sec": round(timestamps[i], 2), "content_type": "video_frame"} for i in range(total)]
    documents = [f"Frame at {timestamps[i]:.1f}s" for i in range(total)]

    client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
    clip_fn = get_clip_text_embedding_function()
    collection = client.get_or_create_collection(name=collection_name, embedding_function=clip_fn)
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
    logger.info(f"Added {total} video frame embeddings to collection {collection_name}")
    return collection


def add_frames_to_existing_collection(
    collection: Any,
    video_path: str,
    fps: float = 1.0,
    max_frames: int = 100,
    progress_callback=None,
) -> int:
    """
    Extract frames from video, embed with CLIP, add to an existing ChromaDB collection.
    Collection must already use CLIP (e.g. for unified video = transcript + frames).
    Returns the number of frames added.
    """
    # #region agent log
    try:
        _lp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cursor", "debug.log")
        open(_lp, "a").write(__import__("json").dumps({"hypothesisId": "H3", "location": "video_vision.py:add_frames_entry", "message": "add_frames_to_existing_collection entry", "data": {"video_path": video_path, "fps": fps, "max_frames": max_frames}, "timestamp": int(__import__("time").time() * 1000), "runId": "video_workflow"}) + "\n")
    except Exception:
        pass
    # #endregion
    try:
        frames_with_ts = extract_frames(video_path, fps=fps, max_frames=max_frames)
        # #region agent log
        try:
            _lp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cursor", "debug.log")
            open(_lp, "a").write(__import__("json").dumps({"hypothesisId": "H3", "location": "video_vision.py:add_frames_after_extract", "message": "after extract_frames", "data": {"num_frames": len(frames_with_ts)}, "timestamp": int(__import__("time").time() * 1000), "runId": "video_workflow"}) + "\n")
        except Exception:
            pass
        # #endregion
        if not frames_with_ts:
            raise RuntimeError("No frames extracted from video")

        frame_bytes = [b for _, b in frames_with_ts]
        timestamps = [t for t, _ in frames_with_ts]
        total = len(frame_bytes)

        if progress_callback:
            progress_callback(0, total)
        embeddings = embed_frames(frame_bytes)
        if progress_callback:
            progress_callback(total, total)

        ids = [f"frame_{i}_{hashlib.sha256(frame_bytes[i]).hexdigest()[:12]}" for i in range(total)]
        metadatas = [{"frame_index": i, "time_sec": round(timestamps[i], 2), "content_type": "video_frame"} for i in range(total)]
        documents = [f"Frame at {timestamps[i]:.1f}s" for i in range(total)]

        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        logger.info(f"Added {total} video frame embeddings to existing collection")
        # #region agent log
        try:
            _lp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cursor", "debug.log")
            open(_lp, "a").write(__import__("json").dumps({"hypothesisId": "H3", "location": "video_vision.py:add_frames_ok", "message": "add_frames_to_existing_collection success", "data": {"total": total}, "timestamp": int(__import__("time").time() * 1000), "runId": "video_workflow"}) + "\n")
        except Exception:
            pass
        # #endregion
        return total
    except Exception as e:
        # #region agent log
        try:
            _lp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cursor", "debug.log")
            open(_lp, "a").write(__import__("json").dumps({"hypothesisId": "H3", "location": "video_vision.py:add_frames_except", "message": "add_frames_to_existing_collection exception", "data": {"type": type(e).__name__, "message": str(e)}, "timestamp": int(__import__("time").time() * 1000), "runId": "video_workflow"}) + "\n")
        except Exception:
            pass
        # #endregion
        raise
