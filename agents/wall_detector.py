"""Wall detector using OpenCV Canny edge detection and Probabilistic Hough Lines."""

import math
import logging
from typing import List, Tuple

import cv2
import numpy as np

from models.floor_plan import WallSegment

logger = logging.getLogger(__name__)


class WallDetector:
    """Detects wall segments from a binary floor plan image."""

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 30,
        hough_min_length: int = 20,
        hough_max_gap: int = 10,
        merge_angle_tolerance: float = 5.0,
        merge_gap_pixels: float = 10.0,
        min_segment_length: int = 20,
        default_wall_thickness: float = 0.2,
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.hough_min_length = hough_min_length
        self.hough_max_gap = hough_max_gap
        self.merge_angle_tolerance = merge_angle_tolerance
        self.merge_gap_pixels = merge_gap_pixels
        self.min_segment_length = min_segment_length
        self.default_wall_thickness = default_wall_thickness

    def detect(
        self, binary_img: np.ndarray, pixels_per_meter: float
    ) -> List[WallSegment]:
        """
        Detect wall segments from a binary image.

        Args:
            binary_img: Binary image (walls = 255/white, background = 0/black)
            pixels_per_meter: Scale factor for converting pixels to meters

        Returns:
            List of WallSegment objects with coordinates in meters
        """
        h, w = binary_img.shape[:2]

        # Apply Canny edge detection on the binary image
        edges = cv2.Canny(binary_img, self.canny_low, self.canny_high)

        # Probabilistic Hough Line Transform
        raw_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_length,
            maxLineGap=self.hough_max_gap,
        )

        if raw_lines is None:
            logger.warning("No lines detected by Hough transform.")
            return []

        segments_px = [(int(l[0][0]), int(l[0][1]), int(l[0][2]), int(l[0][3]))
                       for l in raw_lines]
        logger.debug(f"Hough detected {len(segments_px)} raw line segments.")

        # Filter by minimum length
        segments_px = [s for s in segments_px if self._length(s) >= self.min_segment_length]
        logger.debug(f"{len(segments_px)} segments after length filter.")

        # Merge nearby collinear segments
        merged = self._merge_segments(segments_px)
        logger.debug(f"{len(merged)} segments after merging.")

        # Convert to WallSegment objects in meters
        walls = []
        img_bbox = (0, 0, w, h)
        for seg in merged:
            x1_m = seg[0] / pixels_per_meter
            y1_m = seg[1] / pixels_per_meter
            x2_m = seg[2] / pixels_per_meter
            y2_m = seg[3] / pixels_per_meter
            is_ext = self._is_exterior(seg, img_bbox, tolerance=15)
            wall = WallSegment(
                x1=x1_m,
                y1=y1_m,
                x2=x2_m,
                y2=y2_m,
                thickness=self.default_wall_thickness,
                is_exterior=is_ext,
            )
            walls.append(wall)

        logger.info(f"Detected {len(walls)} wall segments.")
        return walls

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _length(seg: Tuple[int, int, int, int]) -> float:
        dx = seg[2] - seg[0]
        dy = seg[3] - seg[1]
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _angle_deg(seg: Tuple[int, int, int, int]) -> float:
        """Angle in [0, 180) degrees."""
        dx = seg[2] - seg[0]
        dy = seg[3] - seg[1]
        angle = math.degrees(math.atan2(dy, dx)) % 180
        return angle

    def _merge_segments(
        self, segments: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Group segments by angle (±5°) and merge collinear ones that are close.
        """
        if not segments:
            return []

        # Build angle groups
        groups: List[List[Tuple[int, int, int, int]]] = []
        used = [False] * len(segments)

        for i, seg_i in enumerate(segments):
            if used[i]:
                continue
            group = [seg_i]
            used[i] = True
            angle_i = self._angle_deg(seg_i)
            for j, seg_j in enumerate(segments):
                if used[j]:
                    continue
                angle_j = self._angle_deg(seg_j)
                diff = abs(angle_i - angle_j) % 180
                if diff > 90:
                    diff = 180 - diff
                if diff <= self.merge_angle_tolerance:
                    group.append(seg_j)
                    used[j] = True
            groups.append(group)

        merged_all = []
        for group in groups:
            merged_all.extend(self._merge_collinear_group(group))
        return merged_all

    def _merge_collinear_group(
        self, group: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Within a group of similarly-angled segments, merge those that are
        spatially close and collinear (perpendicular distance < merge_gap_pixels).
        """
        if len(group) == 1:
            return group

        result = list(group)
        changed = True
        while changed:
            changed = False
            new_result = []
            used = [False] * len(result)
            for i in range(len(result)):
                if used[i]:
                    continue
                merged_seg = result[i]
                for j in range(i + 1, len(result)):
                    if used[j]:
                        continue
                    candidate = self._try_merge(merged_seg, result[j])
                    if candidate is not None:
                        merged_seg = candidate
                        used[j] = True
                        changed = True
                new_result.append(merged_seg)
                used[i] = True
            result = new_result
        return result

    def _try_merge(
        self,
        a: Tuple[int, int, int, int],
        b: Tuple[int, int, int, int],
    ) -> "Tuple[int,int,int,int] | None":
        """
        Try to merge two segments. Returns merged segment if they are
        close enough, otherwise None.
        """
        # Perpendicular distance from b's midpoint to line through a
        mx = (b[0] + b[2]) / 2.0
        my = (b[1] + b[3]) / 2.0
        perp = self._perp_dist_point_to_segment(mx, my, a)
        if perp > self.merge_gap_pixels:
            return None

        # Check that the gap along the line direction is not too large
        gap = self._gap_between_segments(a, b)
        if gap > self.merge_gap_pixels * 3:
            return None

        # Merge: take the two endpoints that maximise length
        pts = [(a[0], a[1]), (a[2], a[3]), (b[0], b[1]), (b[2], b[3])]
        best_dist = -1.0
        best_pair = (pts[0], pts[1])
        for pi in range(len(pts)):
            for pj in range(pi + 1, len(pts)):
                d = math.dist(pts[pi], pts[pj])
                if d > best_dist:
                    best_dist = d
                    best_pair = (pts[pi], pts[pj])
        return (
            int(best_pair[0][0]),
            int(best_pair[0][1]),
            int(best_pair[1][0]),
            int(best_pair[1][1]),
        )

    @staticmethod
    def _perp_dist_point_to_segment(
        px: float, py: float, seg: Tuple[int, int, int, int]
    ) -> float:
        """Perpendicular distance from point (px,py) to the line through seg."""
        x1, y1, x2, y2 = seg
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            return math.dist((px, py), (x1, y1))
        t = ((px - x1) * dx + (py - y1) * dy) / length_sq
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.dist((px, py), (proj_x, proj_y))

    @staticmethod
    def _gap_between_segments(
        a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
    ) -> float:
        """Approximate gap between two segments (minimum endpoint distance)."""
        pts_a = [(a[0], a[1]), (a[2], a[3])]
        pts_b = [(b[0], b[1]), (b[2], b[3])]
        min_gap = float("inf")
        for pa in pts_a:
            for pb in pts_b:
                min_gap = min(min_gap, math.dist(pa, pb))
        return min_gap

    @staticmethod
    def _is_exterior(
        seg: Tuple[int, int, int, int],
        img_bbox: Tuple[int, int, int, int],
        tolerance: int = 15,
    ) -> bool:
        """
        Determine if a segment is on the exterior boundary (near image edge).
        img_bbox = (x_min, y_min, x_max, y_max)
        """
        x_min, y_min, x_max, y_max = img_bbox
        x1, y1, x2, y2 = seg
        near_left = x1 <= x_min + tolerance and x2 <= x_min + tolerance
        near_right = x1 >= x_max - tolerance and x2 >= x_max - tolerance
        near_top = y1 <= y_min + tolerance and y2 <= y_min + tolerance
        near_bottom = y1 >= y_max - tolerance and y2 >= y_max - tolerance
        return near_left or near_right or near_top or near_bottom
