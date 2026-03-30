"""
Emotion classifier for avatar expressions (NANO-094).

Analyzes LLM response text and produces a mood classification + confidence
score. The mood string maps directly to avatar expression presets (NANO-092).
Classification results are emitted to:
  1. The avatar via AvatarMoodEvent (NANO-093 bridge)
  2. The chat UI via ResponseReadyEvent metadata (display-only, never sent to LLM)

Uses DistilBERT GoEmotions via ONNX Runtime — the same model SillyTavern uses.
~67MB, <15ms CPU inference, no GPU required.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# GoEmotions label order (matches the ONNX model's output indices)
GOEMOTION_LABELS: list[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

# GoEmotions → 5 broad emotion buckets
# Summing probabilities across related labels concentrates the signal
# for mixed-register AI persona text (where no single GoEmotions label dominates)
GOEMOTION_TO_BUCKET: dict[str, Optional[str]] = {
    "admiration": "happy",
    "amusement": "happy",
    "approval": "happy",
    "caring": "happy",
    "excitement": "happy",
    "gratitude": "happy",
    "joy": "happy",
    "love": "happy",
    "optimism": "happy",
    "pride": "happy",
    "relief": "happy",
    "disappointment": "sad",
    "embarrassment": "sad",
    "grief": "sad",
    "remorse": "sad",
    "sadness": "sad",
    "anger": "angry",
    "annoyance": "angry",
    "disapproval": "angry",
    "disgust": "angry",
    "confusion": "surprised",
    "curiosity": "surprised",
    "desire": "surprised",
    "realization": "surprised",
    "surprise": "surprised",
    "fear": "surprised",
    "nervousness": "surprised",
    "neutral": "neutral",
}

# Neutral is 1 label vs 4-11 labels per other bucket — boost its score
# so it can compete fairly. A neutral probability of ~6% becomes ~30% after boost.
_NEUTRAL_WEIGHT = 5.0

# Broad bucket → avatar mood preset (NANO-092)
# neutral → None means no mood change (avatar holds current expression)
BUCKET_TO_MOOD: dict[str, Optional[str]] = {
    "happy": "amused",
    "sad": "melancholy",
    "angry": "annoyed",
    "surprised": "curious",
    "neutral": None,
}

# Legacy export — kept for test compatibility
GOEMOTION_TO_MOOD = GOEMOTION_TO_BUCKET

# HuggingFace model repo for auto-download
_HF_REPO = "Cohee/distilbert-base-uncased-go-emotions-onnx"
_MODEL_FILENAME = "model_quantized.onnx"
_MODEL_HF_PATH = "onnx/model_quantized.onnx"
_TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class ONNXEmotionClassifier:
    """
    DistilBERT GoEmotions classifier via ONNX Runtime.

    Lazy-initializes on first classify() call — does not load the model
    or import onnxruntime at construction time.
    """

    def __init__(
        self,
        model_dir: str = "models/emotion",
        confidence_threshold: float = 0.3,
    ):
        self._model_dir = Path(model_dir)
        self._confidence_threshold = confidence_threshold
        self._session = None  # onnxruntime.InferenceSession (lazy)
        self._tokenizer = None  # tokenizers.Tokenizer (lazy)
        self._initialized = False
        self._disabled = False  # Set True if init fails — don't retry

    def _ensure_model_files(self) -> bool:
        """Download model + tokenizer files from HuggingFace if missing. Returns True on success."""
        self._model_dir.mkdir(parents=True, exist_ok=True)

        # Build map of local_filename → HF repo path
        # Model is in onnx/ subdirectory on HF, tokenizer files are at root
        needed: dict[str, str] = {}
        if not (self._model_dir / _MODEL_FILENAME).exists():
            needed[_MODEL_FILENAME] = _MODEL_HF_PATH
        for f in _TOKENIZER_FILES:
            if not (self._model_dir / f).exists():
                needed[f] = f  # same name on HF

        if not needed:
            return True

        logger.info(
            "Emotion classifier: downloading %d file(s) from %s to %s",
            len(needed), _HF_REPO, self._model_dir,
        )

        try:
            from huggingface_hub import hf_hub_download

            for local_name, hf_path in needed.items():
                path = hf_hub_download(
                    repo_id=_HF_REPO,
                    filename=hf_path,
                )
                # hf_hub_download returns the cached file path — copy to our model dir
                import shutil
                shutil.copy2(path, self._model_dir / local_name)
            logger.info("Emotion classifier: model download complete")
            return True
        except ImportError:
            # huggingface_hub not installed — try raw HTTP
            logger.info("huggingface_hub not installed, trying direct download")
            return self._download_direct(needed)
        except Exception as e:
            logger.error("Emotion classifier: model download failed: %s", e)
            return False

    def _download_direct(self, needed: dict[str, str]) -> bool:
        """Download files directly from HuggingFace CDN without huggingface_hub."""
        import requests

        base_url = f"https://huggingface.co/{_HF_REPO}/resolve/main"
        try:
            for local_name, hf_path in needed.items():
                url = f"{base_url}/{hf_path}"
                logger.info("Downloading %s", url)
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                (self._model_dir / local_name).write_bytes(resp.content)
            logger.info("Emotion classifier: direct download complete")
            return True
        except Exception as e:
            logger.error("Emotion classifier: direct download failed: %s", e)
            return False

    def _init(self) -> bool:
        """Lazy-initialize the ONNX session and tokenizer. Returns True on success."""
        if self._initialized:
            return True
        if self._disabled:
            return False

        # Ensure model files exist (download if needed)
        if not self._ensure_model_files():
            logger.warning("Emotion classifier: model files unavailable — disabling")
            self._disabled = True
            return False

        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            model_path = str(self._model_dir / _MODEL_FILENAME)
            tokenizer_path = str(self._model_dir / "tokenizer.json")

            # ONNX session — CPU only, minimal threads
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._session = ort.InferenceSession(
                model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )

            # Tokenizer from tokenizers library (not transformers — lighter)
            self._tokenizer = Tokenizer.from_file(tokenizer_path)
            self._tokenizer.enable_truncation(max_length=512)
            self._tokenizer.enable_padding(length=512)

            self._initialized = True
            logger.info("Emotion classifier initialized (ONNX + tokenizer loaded)")
            return True

        except Exception as e:
            logger.error("Emotion classifier: initialization failed: %s", e)
            self._disabled = True
            return False

    def classify(self, text: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Classify text and return (avatar_mood, confidence).

        Returns (None, None) if:
        - Classifier is disabled or failed to init
        - Text is empty
        - Top label confidence is below threshold
        - Top label maps to None (neutral)
        """
        if not text or not text.strip():
            return None, None

        if not self._init():
            return None, None

        try:
            # Tokenize
            encoding = self._tokenizer.encode(text)
            input_ids = np.array([encoding.ids], dtype=np.int64)
            attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

            # Infer
            outputs = self._session.run(
                None,
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )

            # outputs[0] shape: (1, 28) — logits
            logits = outputs[0]
            probs = _softmax(logits)[0]

            # Sum probabilities into broad buckets (happy/sad/angry/surprised/neutral)
            # This concentrates signal for mixed-register AI text
            bucket_scores: dict[str, float] = {}
            for i, label in enumerate(GOEMOTION_LABELS):
                bucket = GOEMOTION_TO_BUCKET.get(label)
                if bucket is not None:
                    score = float(probs[i])
                    # Neutral is 1 label vs 4-11 per bucket — weight it up
                    if bucket == "neutral":
                        score *= _NEUTRAL_WEIGHT
                    bucket_scores[bucket] = bucket_scores.get(bucket, 0.0) + score

            if not bucket_scores:
                return None, None

            # Normalize bucket scores to [0, 1] (true probability distribution)
            total = sum(bucket_scores.values())
            if total > 0:
                bucket_scores = {k: v / total for k, v in bucket_scores.items()}

            # Top bucket
            top_bucket = max(bucket_scores, key=bucket_scores.get)
            confidence = bucket_scores[top_bucket]

            if confidence < self._confidence_threshold:
                return None, None

            mood = BUCKET_TO_MOOD.get(top_bucket)
            if mood is None:
                return None, None

            return mood, round(confidence, 4)

        except Exception as e:
            logger.error("Emotion classifier: inference failed: %s", e)
            return None, None
