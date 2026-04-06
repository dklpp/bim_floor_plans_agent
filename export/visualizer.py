"""Visualizer: matplotlib 2D and 3D previews of a FloorPlan."""

import logging
import math
import os
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import numpy as np

from models.floor_plan import FloorPlan

logger = logging.getLogger(__name__)

# Use non-interactive backend when no display is available
matplotlib.use("Agg")


class Visualizer:
    """Generates 2D and 3D visualizations of a parsed FloorPlan."""

    # Colour palette for rooms
    ROOM_COLORS = [
        "#AED6F1",  # light blue
        "#A9DFBF",  # light green
        "#F9E79F",  # light yellow
        "#F5CBA7",  # light orange
        "#D2B4DE",  # light purple
        "#FADBD8",  # light pink
        "#D5DBDB",  # light grey
        "#A3E4D7",  # mint
    ]

    def plot_floor_plan(
        self,
        floor_plan: FloorPlan,
        output_path: Optional[str] = None,
        show: bool = True,
    ) -> Optional[str]:
        """
        Create a 2D matplotlib plot of the floor plan.

        Rooms are drawn as filled polygons, walls as thick lines,
        doors and windows as markers.

        Args:
            floor_plan: The parsed floor plan.
            output_path: If provided, save the figure to this path.
            show: Whether to call plt.show() (requires a display).

        Returns:
            Path where the figure was saved, or None.
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_aspect("equal")
        ax.set_title("Floor Plan (2D)", fontsize=14, fontweight="bold")
        ax.set_xlabel("X (metres)")
        ax.set_ylabel("Y (metres)")

        # Draw rooms as filled polygons
        for idx, room in enumerate(floor_plan.rooms):
            if len(room.polygon) < 3:
                continue
            color = self.ROOM_COLORS[idx % len(self.ROOM_COLORS)]
            poly = MplPolygon(room.polygon, closed=True, facecolor=color, alpha=0.6, edgecolor="none")
            ax.add_patch(poly)
            # Label at centroid
            cx = sum(p[0] for p in room.polygon) / len(room.polygon)
            cy = sum(p[1] for p in room.polygon) / len(room.polygon)
            ax.text(
                cx, cy,
                f"{room.name}\n{room.area:.1f} m²",
                ha="center", va="center", fontsize=7, color="#1A1A1A",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"),
            )

        # Draw walls
        for wall in floor_plan.walls:
            color = "#1A1A1A" if wall.is_exterior else "#555555"
            lw = 2.5 if wall.is_exterior else 1.5
            ax.plot([wall.x1, wall.x2], [wall.y1, wall.y2], color=color, linewidth=lw)

        # Draw doors
        for door in floor_plan.doors:
            ax.plot(door.x, door.y, marker="D", color="saddlebrown", markersize=8)
            ax.annotate("D", (door.x, door.y), textcoords="offset points",
                        xytext=(5, 5), fontsize=6, color="saddlebrown")

        # Draw windows
        for win in floor_plan.windows:
            ax.plot(win.x, win.y, marker="s", color="steelblue", markersize=8)
            ax.annotate("W", (win.x, win.y), textcoords="offset points",
                        xytext=(5, 5), fontsize=6, color="steelblue")

        # Legend
        legend_handles = [
            mpatches.Patch(color="#1A1A1A", label="Exterior wall"),
            mpatches.Patch(color="#555555", label="Interior wall"),
        ]
        if floor_plan.doors:
            legend_handles.append(
                mpatches.Patch(color="saddlebrown", label="Door")
            )
        if floor_plan.windows:
            legend_handles.append(
                mpatches.Patch(color="steelblue", label="Window")
            )
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

        # Axis limits with margin
        margin = 0.5
        ax.set_xlim(-margin, floor_plan.width_m + margin)
        ax.set_ylim(-margin, floor_plan.height_m + margin)

        plt.tight_layout()

        saved_path = None
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            saved_path = os.path.abspath(output_path)
            logger.info(f"2D visualization saved: {saved_path}")

        if show:
            try:
                plt.show()
            except Exception:
                pass

        plt.close(fig)
        return saved_path

    def plot_3d_preview(
        self,
        floor_plan: FloorPlan,
        ceiling_height: float = 2.8,
        output_path: Optional[str] = None,
        show: bool = True,
    ) -> Optional[str]:
        """
        Create a simple 3D matplotlib visualization of the floor plan.

        Each wall segment is drawn as a vertical rectangular box.

        Args:
            floor_plan: The parsed floor plan.
            ceiling_height: Height of walls in metres.
            output_path: If provided, save the figure to this path.
            show: Whether to call plt.show().

        Returns:
            Path where the figure was saved, or None.
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Floor Plan (3D Preview)", fontsize=14, fontweight="bold")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        # Draw floor plane
        fw = floor_plan.width_m
        fh = floor_plan.height_m
        floor_verts = [[(0, 0, 0), (fw, 0, 0), (fw, fh, 0), (0, fh, 0)]]
        floor_poly = Poly3DCollection(floor_verts, alpha=0.15, facecolor="#E8E8E8", edgecolor="#AAAAAA")
        ax.add_collection3d(floor_poly)

        # Draw walls as 3D boxes
        for wall in floor_plan.walls:
            verts = self._wall_to_3d_verts(wall, ceiling_height)
            face_color = "#C0C0C0" if wall.is_exterior else "#D8D8D8"
            edge_color = "#404040" if wall.is_exterior else "#666666"
            poly = Poly3DCollection(
                verts, alpha=0.85, facecolor=face_color, edgecolor=edge_color, linewidth=0.5
            )
            ax.add_collection3d(poly)

        # Axis limits
        ax.set_xlim(0, fw)
        ax.set_ylim(0, fh)
        ax.set_zlim(0, ceiling_height * 1.1)

        ax.view_init(elev=30, azim=-60)
        plt.tight_layout()

        saved_path = None
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            saved_path = os.path.abspath(output_path)
            logger.info(f"3D visualization saved: {saved_path}")

        if show:
            try:
                plt.show()
            except Exception:
                pass

        plt.close(fig)
        return saved_path

    def plot_top_view_3d(
        self,
        floor_plan: FloorPlan,
        ceiling_height: float = 2.8,
        output_path: Optional[str] = None,
        show: bool = False,
        tilt: float = 65.0,
    ) -> Optional[str]:
        """
        Top-down 3D floor plan: rooms as coloured floor polygons, walls extruded.

        Args:
            floor_plan:     Parsed floor plan model.
            ceiling_height: Wall height in metres.
            output_path:    Save path for the PNG.
            show:           Whether to call plt.show().
            tilt:           Elevation angle in degrees (90 = pure top-down,
                            65-75 gives a slight 3-D feel while keeping rooms readable).
        """
        from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.patches import Patch

        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Floor Plan — Top View (3D)", fontsize=14, fontweight="bold")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        fw = floor_plan.width_m
        fh = floor_plan.height_m

        # ── Floor slab ──────────────────────────────────────────────────
        floor_verts = [[(0, 0, 0), (fw, 0, 0), (fw, fh, 0), (0, fh, 0)]]
        ax.add_collection3d(
            Poly3DCollection(floor_verts, alpha=0.08,
                             facecolor="#CCCCCC", edgecolor="#AAAAAA", linewidth=0.5)
        )

        # ── Room polygons on the floor plane ────────────────────────────
        legend_handles = []
        for idx, room in enumerate(floor_plan.rooms):
            if len(room.polygon) < 3:
                continue
            color = self.ROOM_COLORS[idx % len(self.ROOM_COLORS)]
            # Close the polygon
            pts_3d = [(x, y, 0.01) for x, y in room.polygon]
            ax.add_collection3d(
                Poly3DCollection([pts_3d], alpha=0.55,
                                 facecolor=color, edgecolor="none")
            )
            # Centroid label
            cx = sum(p[0] for p in room.polygon) / len(room.polygon)
            cy = sum(p[1] for p in room.polygon) / len(room.polygon)
            ax.text(cx, cy, 0.05,
                    f"{room.name}\n{room.area:.1f} m²",
                    ha="center", va="center", fontsize=6.5,
                    color="#111111",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              alpha=0.7, edgecolor="none"))
            legend_handles.append(Patch(facecolor=color, alpha=0.7, label=f"{room.name}"))

        # ── Walls ────────────────────────────────────────────────────────
        for wall in floor_plan.walls:
            verts = self._wall_to_3d_verts(wall, ceiling_height)
            if not verts:
                continue
            face_color = "#5A5A5A" if wall.is_exterior else "#888888"
            edge_color = "#333333" if wall.is_exterior else "#555555"
            lw = 0.4 if wall.is_exterior else 0.3
            ax.add_collection3d(
                Poly3DCollection(verts, alpha=0.90,
                                 facecolor=face_color,
                                 edgecolor=edge_color,
                                 linewidth=lw)
            )

        # ── Ceiling plane (transparent) ──────────────────────────────────
        ceil_verts = [[(0, 0, ceiling_height), (fw, 0, ceiling_height),
                       (fw, fh, ceiling_height), (0, fh, ceiling_height)]]
        ax.add_collection3d(
            Poly3DCollection(ceil_verts, alpha=0.04,
                             facecolor="#DDDDDD", edgecolor="#BBBBBB", linewidth=0.3)
        )

        # ── Axis limits & camera ─────────────────────────────────────────
        ax.set_xlim(0, fw)
        ax.set_ylim(0, fh)
        ax.set_zlim(0, ceiling_height * 1.05)
        ax.view_init(elev=tilt, azim=-80)

        # Legend (rooms only, capped at 8 entries to avoid clutter)
        if legend_handles:
            ax.legend(handles=legend_handles[:8], loc="upper right",
                      fontsize=7, framealpha=0.8)

        plt.tight_layout()

        saved_path = None
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            saved_path = os.path.abspath(output_path)
            logger.info(f"Top-view 3D saved: {saved_path}")

        if show:
            try:
                plt.show()
            except Exception:
                pass

        plt.close(fig)
        return saved_path

    @staticmethod
    def _wall_to_3d_verts(wall, ceiling_height: float):
        """
        Convert a WallSegment to a list of 3D polygon faces for Poly3DCollection.
        Returns 6 faces (box) but we draw just the 4 visible sides + top.
        """
        x1, y1, x2, y2 = wall.x1, wall.y1, wall.x2, wall.y2
        t = wall.thickness / 2.0

        # Perpendicular offset direction
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-9:
            return []
        nx = -dy / length * t
        ny = dx / length * t

        # 8 corners of the wall box
        # Bottom face
        b0 = (x1 + nx, y1 + ny, 0.0)
        b1 = (x2 + nx, y2 + ny, 0.0)
        b2 = (x2 - nx, y2 - ny, 0.0)
        b3 = (x1 - nx, y1 - ny, 0.0)
        # Top face
        h = ceiling_height
        t0 = (x1 + nx, y1 + ny, h)
        t1 = (x2 + nx, y2 + ny, h)
        t2 = (x2 - nx, y2 - ny, h)
        t3 = (x1 - nx, y1 - ny, h)

        faces = [
            [b0, b1, t1, t0],   # front face
            [b3, b2, t2, t3],   # back face
            [b0, b3, t3, t0],   # left end
            [b1, b2, t2, t1],   # right end
            [t0, t1, t2, t3],   # top face
        ]
        return faces
