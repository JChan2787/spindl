"""
ScreenCapture - MSS-based screen capture with preprocessing.

Captures the screen, resizes to target dimensions, and encodes as JPEG
for transmission to VLM endpoints.
"""

import base64
import io
import logging
from dataclasses import dataclass
from typing import Optional

import mss
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class CaptureResult:
    """
    Result from a screen capture operation.

    Contains the image data in multiple formats for flexibility.
    """

    jpeg_bytes: bytes
    """Raw JPEG-encoded image bytes."""

    base64_string: str
    """Base64-encoded JPEG string for API transmission."""

    width: int
    """Actual width of captured image after resize."""

    height: int
    """Actual height of captured image after resize."""

    original_width: int
    """Original screen width before resize."""

    original_height: int
    """Original screen height before resize."""


class ScreenCapture:
    """
    MSS-based screen capture with preprocessing.

    Captures the specified monitor, resizes to target dimensions using
    high-quality LANCZOS resampling, and encodes as JPEG for efficient
    transmission to VLM endpoints.

    Usage:
        capture = ScreenCapture(monitor=1, target_width=1920, target_height=1080)
        result = capture.capture()
        # Use result.base64_string for API calls
    """

    def __init__(
        self,
        monitor: int = 1,
        target_width: int = 1920,
        target_height: int = 1080,
        jpeg_quality: int = 95,
    ):
        """
        Initialize screen capture with target dimensions.

        Args:
            monitor: Monitor index (1 = primary, 2 = secondary, etc.)
                    0 captures all monitors as one virtual screen.
            target_width: Target width after resize (pixels)
            target_height: Target height after resize (pixels)
            jpeg_quality: JPEG encoding quality (1-100, higher = better quality)
        """
        self.monitor = monitor
        self.target_width = target_width
        self.target_height = target_height
        self.jpeg_quality = jpeg_quality
        self._sct: Optional[mss.mss] = None

    def _get_mss(self) -> mss.mss:
        """
        Create a fresh MSS instance for each capture.

        MSS instances are NOT thread-safe on Windows - the srcdc (source
        device context) is stored in thread-local storage. Caching the
        instance causes 'srcdc' attribute errors when called from different
        threads. Creating fresh instances is slightly slower but reliable.
        """
        # Always create fresh instance - mss is not thread-safe on Windows
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
        self._sct = mss.mss()
        return self._sct

    def capture(self) -> CaptureResult:
        """
        Capture screen, resize, and return as CaptureResult.

        Returns:
            CaptureResult containing JPEG bytes and base64 string

        Raises:
            RuntimeError: If capture fails (invalid monitor, etc.)
        """
        sct = self._get_mss()

        # Validate monitor index
        if self.monitor >= len(sct.monitors):
            raise RuntimeError(
                f"Monitor {self.monitor} not found. "
                f"Available: 0-{len(sct.monitors) - 1}"
            )

        # Capture the screen
        monitor_info = sct.monitors[self.monitor]
        screenshot = sct.grab(monitor_info)

        original_width = screenshot.width
        original_height = screenshot.height

        # Convert to PIL Image
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

        # Resize with high-quality resampling
        img = img.resize(
            (self.target_width, self.target_height),
            Image.Resampling.LANCZOS,
        )

        # Encode as JPEG
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.jpeg_quality)
        jpeg_bytes = buffer.getvalue()

        # Base64 encode for API transmission
        base64_string = base64.b64encode(jpeg_bytes).decode("utf-8")

        logger.debug(
            f"Captured screen: {original_width}x{original_height} -> "
            f"{self.target_width}x{self.target_height}, "
            f"{len(jpeg_bytes)} bytes"
        )

        return CaptureResult(
            jpeg_bytes=jpeg_bytes,
            base64_string=base64_string,
            width=self.target_width,
            height=self.target_height,
            original_width=original_width,
            original_height=original_height,
        )

    def capture_bytes(self) -> bytes:
        """
        Capture screen and return raw JPEG bytes.

        Convenience method for when you only need the bytes.

        Returns:
            JPEG-encoded image bytes
        """
        return self.capture().jpeg_bytes

    def capture_base64(self) -> str:
        """
        Capture screen and return base64-encoded JPEG string.

        Convenience method for direct API transmission.

        Returns:
            Base64-encoded JPEG string
        """
        return self.capture().base64_string

    def list_monitors(self) -> list[dict]:
        """
        List available monitors with their dimensions.

        Useful for configuration and debugging.

        Returns:
            List of monitor info dicts with 'index', 'width', 'height', 'left', 'top'
        """
        sct = self._get_mss()
        monitors = []
        for i, mon in enumerate(sct.monitors):
            monitors.append(
                {
                    "index": i,
                    "width": mon["width"],
                    "height": mon["height"],
                    "left": mon["left"],
                    "top": mon["top"],
                    "is_virtual": i == 0,  # Monitor 0 is virtual "all monitors"
                }
            )
        return monitors

    def close(self) -> None:
        """
        Release MSS resources.

        Called automatically when the ScreenCapture is garbage collected,
        but can be called explicitly for deterministic cleanup.
        """
        if self._sct is not None:
            self._sct.close()
            self._sct = None

    def __enter__(self) -> "ScreenCapture":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup resources."""
        self.close()
