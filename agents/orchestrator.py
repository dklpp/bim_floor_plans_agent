"""BIM Orchestrator: ties together all sub-agents for end-to-end processing."""

import logging
import os
import time
from typing import Dict, Any, Optional

from agents.floor_plan_parser import FloorPlanParser
from agents.wall_detector import WallDetector
from agents.room_segmentor import RoomSegmentor
from export.ifc_exporter import IFCExporter
from export.visualizer import Visualizer
from models.floor_plan import FloorPlan

logger = logging.getLogger(__name__)


class BIMOrchestrator:
    """
    High-level orchestrator that coordinates parsing, detection,
    segmentation, IFC export and visualization.
    """

    def __init__(self):
        self.wall_detector = WallDetector()
        self.room_segmentor = RoomSegmentor()
        self.parser = FloorPlanParser(
            wall_detector=self.wall_detector,
            room_segmentor=self.room_segmentor,
        )
        self.ifc_exporter = IFCExporter()
        self.visualizer = Visualizer()

    def process(
        self,
        image_path: str,
        output_dir: str,
        pixels_per_meter: float = 100.0,
        ceiling_height: float = 2.8,
        show_viz: bool = False,
    ) -> Dict[str, Any]:
        """
        Full processing pipeline: image → parsed floor plan → IFC + visualization.

        Args:
            image_path: Path to the input floor plan image.
            output_dir: Directory to write output files.
            pixels_per_meter: Image scale (pixels per metre).
            ceiling_height: Wall height in metres.
            show_viz: Whether to display matplotlib windows.

        Returns:
            Dictionary with keys:
              - ifc_path: str  — path to the IFC file
              - viz_2d_path: str | None — path to 2D visualization image
              - viz_3d_path: str | None — path to 3D visualization image
              - floor_plan: FloorPlan — the parsed model
              - stats: dict — summary statistics
              - elapsed_s: float — total processing time
        """
        os.makedirs(output_dir, exist_ok=True)
        t_start = time.time()

        # ---- Step 1: Parse floor plan ----
        logger.info("[1/4] Parsing floor plan image...")
        floor_plan: FloorPlan = self.parser.parse(image_path, pixels_per_meter)
        logger.info(
            f"      Done: {len(floor_plan.walls)} walls, {len(floor_plan.rooms)} rooms."
        )

        # ---- Step 2: Export to IFC ----
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        ifc_path = os.path.join(output_dir, f"{base_name}.ifc")
        logger.info(f"[2/4] Exporting IFC → {ifc_path}")
        ifc_path = self.ifc_exporter.export(
            floor_plan,
            ifc_path,
            ceiling_height=ceiling_height,
        )
        logger.info("      IFC export complete.")

        # ---- Step 3: 2D visualization ----
        viz_2d_path: Optional[str] = None
        viz_2d_file = os.path.join(output_dir, f"{base_name}_2d.png")
        logger.info(f"[3/4] Generating 2D visualization → {viz_2d_file}")
        try:
            viz_2d_path = self.visualizer.plot_floor_plan(
                floor_plan, output_path=viz_2d_file, show=show_viz
            )
        except Exception as exc:
            logger.warning(f"      2D visualization failed: {exc}")

        # ---- Step 4: 3D visualization ----
        viz_3d_path: Optional[str] = None
        viz_3d_file = os.path.join(output_dir, f"{base_name}_3d.png")
        logger.info(f"[4/4] Generating 3D visualization → {viz_3d_file}")
        try:
            viz_3d_path = self.visualizer.plot_3d_preview(
                floor_plan,
                ceiling_height=ceiling_height,
                output_path=viz_3d_file,
                show=show_viz,
            )
        except Exception as exc:
            logger.warning(f"      3D visualization failed: {exc}")

        elapsed = time.time() - t_start

        stats = {
            "num_walls": len(floor_plan.walls),
            "num_exterior_walls": sum(1 for w in floor_plan.walls if w.is_exterior),
            "num_interior_walls": sum(1 for w in floor_plan.walls if not w.is_exterior),
            "num_rooms": len(floor_plan.rooms),
            "num_doors": len(floor_plan.doors),
            "num_windows": len(floor_plan.windows),
            "floor_width_m": round(floor_plan.width_m, 2),
            "floor_height_m": round(floor_plan.height_m, 2),
            "total_area_m2": round(
                sum(r.area for r in floor_plan.rooms), 2
            ),
        }

        logger.info(
            f"Processing complete in {elapsed:.2f}s. "
            f"IFC: {ifc_path}"
        )

        return {
            "ifc_path": ifc_path,
            "viz_2d_path": viz_2d_path,
            "viz_3d_path": viz_3d_path,
            "floor_plan": floor_plan,
            "stats": stats,
            "elapsed_s": round(elapsed, 2),
        }
