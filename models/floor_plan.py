"""Dataclasses for representing a 2D floor plan parsed from an image."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class WallSegment:
    """A wall segment with start/end coordinates in meters."""
    x1: float  # Start x in meters
    y1: float  # Start y in meters
    x2: float  # End x in meters
    y2: float  # End y in meters
    thickness: float = 0.2  # Wall thickness in meters
    is_exterior: bool = False  # True if this is an exterior (outer) wall

    @property
    def length(self) -> float:
        """Length of the wall segment in meters."""
        import math
        return math.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)

    @property
    def angle_degrees(self) -> float:
        """Angle of the wall segment in degrees."""
        import math
        return math.degrees(math.atan2(self.y2 - self.y1, self.x2 - self.x1))


@dataclass
class Door:
    """A door element on the floor plan."""
    x: float        # Center x in meters
    y: float        # Center y in meters
    width: float    # Door width in meters
    angle: float    # Orientation angle in degrees


@dataclass
class Window:
    """A window element on the floor plan."""
    x: float        # Center x in meters
    y: float        # Center y in meters
    width: float    # Window width in meters
    angle: float    # Orientation angle in degrees


@dataclass
class Room:
    """A room region detected in the floor plan."""
    polygon: List[Tuple[float, float]]  # List of (x, y) tuples in meters
    name: str = "Room"
    area: float = 0.0  # Area in square meters

    def compute_area(self) -> float:
        """Compute polygon area using the shoelace formula."""
        n = len(self.polygon)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.polygon[i][0] * self.polygon[j][1]
            area -= self.polygon[j][0] * self.polygon[i][1]
        return abs(area) / 2.0


@dataclass
class FloorPlan:
    """Complete floor plan model containing all detected elements."""
    walls: List[WallSegment] = field(default_factory=list)
    doors: List[Door] = field(default_factory=list)
    windows: List[Window] = field(default_factory=list)
    rooms: List[Room] = field(default_factory=list)
    width_m: float = 0.0          # Total width in meters
    height_m: float = 0.0         # Total height in meters
    pixels_per_meter: float = 100.0  # Scale factor used during parsing

    def summary(self) -> str:
        """Return a human-readable summary of the floor plan."""
        lines = [
            f"FloorPlan Summary:",
            f"  Dimensions: {self.width_m:.2f}m x {self.height_m:.2f}m",
            f"  Scale: {self.pixels_per_meter:.1f} px/m",
            f"  Walls: {len(self.walls)} segments "
            f"({sum(1 for w in self.walls if w.is_exterior)} exterior, "
            f"{sum(1 for w in self.walls if not w.is_exterior)} interior)",
            f"  Doors: {len(self.doors)}",
            f"  Windows: {len(self.windows)}",
            f"  Rooms: {len(self.rooms)}",
        ]
        for room in self.rooms:
            lines.append(f"    - {room.name}: {room.area:.2f} m²")
        return "\n".join(lines)
