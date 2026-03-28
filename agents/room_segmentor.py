"""Room segmentor: finds enclosed regions between walls using flood-fill."""

import logging
from typing import List, Tuple

import cv2
import numpy as np

from models.floor_plan import Room, WallSegment

logger = logging.getLogger(__name__)


class RoomSegmentor:
    """Segments rooms from a binary floor plan image using flood-fill."""

    def __init__(
        self,
        min_room_area_px: int = 1000,
        dilation_kernel_size: int = 3,
        room_name_prefix: str = "Room",
    ):
        self.min_room_area_px = min_room_area_px
        self.dilation_kernel_size = dilation_kernel_size
        self.room_name_prefix = room_name_prefix

    def segment(
        self,
        binary_img: np.ndarray,
        walls: List[WallSegment],
        pixels_per_meter: float,
    ) -> List[Room]:
        """
        Find enclosed room regions by flood-filling the spaces between walls.

        Args:
            binary_img: Binary image where walls are white (255) and
                        background is black (0).
            walls: Detected wall segments (unused for flood-fill but kept for
                   future filtering/labelling improvements).
            pixels_per_meter: Scale factor for converting pixels to meters.

        Returns:
            List of Room objects with polygons and areas in meters.
        """
        h, w = binary_img.shape[:2]

        # Dilate the wall image slightly to close small gaps
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.dilation_kernel_size, self.dilation_kernel_size),
        )
        wall_mask = cv2.dilate(binary_img, kernel, iterations=2)

        # Invert: background becomes white, walls become black
        # We will flood-fill the white (background) regions
        inverted = cv2.bitwise_not(wall_mask)

        # Label connected components in the inverted image
        num_labels, label_map, stats, centroids = cv2.connectedComponentsWithStats(
            inverted, connectivity=4
        )

        rooms: List[Room] = []
        room_counter = 1

        for label_idx in range(1, num_labels):  # skip background (label 0)
            area_px = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area_px < self.min_room_area_px:
                continue

            # Extract the region mask
            region_mask = (label_map == label_idx).astype(np.uint8) * 255

            # Find contour of this region
            contours, _ = cv2.findContours(
                region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            # Use the largest contour
            contour = max(contours, key=cv2.contourArea)

            # Approximate contour to reduce vertices
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < 3:
                continue

            # Convert pixel coordinates to meters
            polygon = [
                (float(pt[0][0]) / pixels_per_meter, float(pt[0][1]) / pixels_per_meter)
                for pt in approx
            ]

            area_m2 = area_px / (pixels_per_meter ** 2)

            room = Room(
                polygon=polygon,
                name=f"{self.room_name_prefix}_{room_counter}",
                area=area_m2,
            )
            rooms.append(room)
            room_counter += 1

        # Sort rooms by area descending
        rooms.sort(key=lambda r: r.area, reverse=True)

        # Re-label with descriptive names based on size order
        room_labels = self._assign_room_names(rooms)
        for room, label in zip(rooms, room_labels):
            room.name = label

        logger.info(f"Segmented {len(rooms)} rooms.")
        return rooms

    @staticmethod
    def _assign_room_names(rooms: List[Room]) -> List[str]:
        """
        Assign descriptive names to rooms based on area ranking.
        Largest → Living Room, then Bedroom, Kitchen, Bathroom, etc.
        """
        name_map = [
            "Living Room",
            "Bedroom",
            "Kitchen",
            "Bathroom",
            "Hallway",
            "Study",
            "Dining Room",
            "Storage",
        ]
        names = []
        for i, room in enumerate(rooms):
            if i < len(name_map):
                names.append(name_map[i])
            else:
                names.append(f"Room_{i + 1}")
        return names
