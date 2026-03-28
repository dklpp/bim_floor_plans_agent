"""Floor plan parser: loads an image and orchestrates detection."""

import logging
from typing import Optional

import cv2
import numpy as np

from models.floor_plan import FloorPlan
from agents.wall_detector import WallDetector
from agents.room_segmentor import RoomSegmentor

logger = logging.getLogger(__name__)


class FloorPlanParser:
    """Parses a 2D floor plan image into a structured FloorPlan model."""

    def __init__(
        self,
        wall_detector: Optional[WallDetector] = None,
        room_segmentor: Optional[RoomSegmentor] = None,
    ):
        self.wall_detector = wall_detector or WallDetector()
        self.room_segmentor = room_segmentor or RoomSegmentor()

    def load_image(self, path: str) -> np.ndarray:
        """
        Load an image from disk.

        Args:
            path: Path to the image file.

        Returns:
            numpy array (BGR colour image).
        """
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        logger.debug(f"Loaded image {path} shape={img.shape}")
        return img

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Convert a colour floor plan image into a clean binary image.

        Pipeline:
          1. Convert to grayscale
          2. Gaussian blur to reduce noise
          3. Adaptive threshold to handle uneven lighting
          4. Morphological close to fill small gaps in walls
          5. Morphological open to remove small noise blobs
          6. Invert so walls = 255, background = 0

        Args:
            img: BGR or grayscale numpy array.

        Returns:
            Binary numpy array (walls = 255, background = 0).
        """
        # Grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive threshold (inverts: dark lines become white on black)
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=4,
        )

        # Morphological close: fill small gaps in walls
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=2)

        # Morphological open: remove small noise blobs
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)

        logger.debug(f"Preprocessed image to binary shape={binary.shape}")
        return binary

    def parse(
        self,
        image_path: str,
        pixels_per_meter: float = 100.0,
    ) -> FloorPlan:
        """
        Full parsing pipeline: load → preprocess → detect walls → segment rooms.

        Args:
            image_path: Path to the floor plan image.
            pixels_per_meter: Scale factor (pixels per metre in the image).

        Returns:
            Populated FloorPlan object.
        """
        logger.info(f"Parsing floor plan: {image_path}")

        # Load and preprocess
        img = self.load_image(image_path)
        binary = self.preprocess(img)

        h, w = binary.shape[:2]
        width_m = w / pixels_per_meter
        height_m = h / pixels_per_meter

        # Detect walls
        logger.info("Detecting walls...")
        walls = self.wall_detector.detect(binary, pixels_per_meter)

        # Segment rooms
        logger.info("Segmenting rooms...")
        rooms = self.room_segmentor.segment(binary, walls, pixels_per_meter)

        floor_plan = FloorPlan(
            walls=walls,
            doors=[],
            windows=[],
            rooms=rooms,
            width_m=width_m,
            height_m=height_m,
            pixels_per_meter=pixels_per_meter,
        )

        logger.info(
            f"Parsing complete: {len(walls)} walls, {len(rooms)} rooms."
        )
        return floor_plan
