"""IFC exporter: creates a valid IFC4 file from a FloorPlan model."""

import math
import logging
import os
from typing import Tuple

import ifcopenshell  # type: ignore
import ifcopenshell.api  # type: ignore
import ifcopenshell.api.root  # type: ignore
import ifcopenshell.api.unit  # type: ignore
import ifcopenshell.api.context  # type: ignore
import ifcopenshell.api.project  # type: ignore
import ifcopenshell.api.geometry  # type: ignore
import ifcopenshell.api.aggregate  # type: ignore
import ifcopenshell.api.spatial  # type: ignore

from models.floor_plan import FloorPlan, WallSegment

logger = logging.getLogger(__name__)


def _create_wall_geometry(
    model: ifcopenshell.file,
    body_context,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    thickness: float,
    height: float,
) -> Tuple:
    """
    Create IFC geometry (IfcExtrudedAreaSolid) for a wall segment.

    The wall is placed at its start point (x1, y1), oriented along the
    direction (x2-x1, y2-y1), with a rectangular cross-section
    (length × thickness) extruded to the given height.

    Returns:
        (extrusion, placement_3d) tuple.
    """
    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if length < 1e-6:
        raise ValueError(f"Wall has zero length: ({x1},{y1})→({x2},{y2})")

    angle = math.atan2(y2 - y1, x2 - x1)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # --- Wall local placement (at start point) ---
    origin = model.createIfcCartesianPoint([float(x1), float(y1), 0.0])
    x_dir = model.createIfcDirection([cos_a, sin_a, 0.0])
    z_dir = model.createIfcDirection([0.0, 0.0, 1.0])
    placement_3d = model.createIfcAxis2Placement3D(origin, z_dir, x_dir)

    # --- Rectangle profile centred along the wall length ---
    profile_origin = model.createIfcCartesianPoint([length / 2.0, 0.0])
    profile_x_dir = model.createIfcDirection([1.0, 0.0])
    profile_placement = model.createIfcAxis2Placement2D(profile_origin, profile_x_dir)
    rect_profile = model.createIfcRectangleProfileDef(
        "AREA", None, profile_placement, float(length), float(thickness)
    )

    # --- Extrusion placement (identity, extrudes along Z) ---
    extrusion_origin = model.createIfcCartesianPoint([0.0, 0.0, 0.0])
    extrusion_z = model.createIfcDirection([0.0, 0.0, 1.0])
    extrusion_x = model.createIfcDirection([1.0, 0.0, 0.0])
    extrusion_placement = model.createIfcAxis2Placement3D(
        extrusion_origin, extrusion_z, extrusion_x
    )

    extrusion_dir = model.createIfcDirection([0.0, 0.0, 1.0])
    extrusion = model.createIfcExtrudedAreaSolid(
        rect_profile, extrusion_placement, extrusion_dir, float(height)
    )
    return extrusion, placement_3d


class IFCExporter:
    """Exports a FloorPlan to a valid IFC4 file using ifcopenshell."""

    def export(
        self,
        floor_plan: FloorPlan,
        output_path: str,
        ceiling_height: float = 2.8,
        storey_name: str = "Ground Floor",
    ) -> str:
        """
        Create an IFC4 file from the parsed floor plan.

        Args:
            floor_plan: The parsed FloorPlan model.
            output_path: Path where the .ifc file will be written.
            ceiling_height: Floor-to-ceiling height in metres.
            storey_name: Name of the building storey.

        Returns:
            Absolute path to the created IFC file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        logger.info(f"Creating IFC model at {output_path}")

        # ------------------------------------------------------------------
        # Initialise model and project hierarchy
        # ------------------------------------------------------------------
        model = ifcopenshell.file(schema="IFC4")

        project = ifcopenshell.api.run(
            "root.create_entity", model, ifc_class="IfcProject", name="BIM Project"
        )
        ifcopenshell.api.run("unit.assign_unit", model)

        # Representation contexts
        context = ifcopenshell.api.run(
            "context.add_context", model, context_type="Model"
        )
        body_context = ifcopenshell.api.run(
            "context.add_context",
            model,
            context_type="Model",
            context_identifier="Body",
            target_view="MODEL_VIEW",
            parent=context,
        )

        # Site / Building / Storey hierarchy
        site = ifcopenshell.api.run(
            "root.create_entity", model, ifc_class="IfcSite", name="Site"
        )
        building = ifcopenshell.api.run(
            "root.create_entity", model, ifc_class="IfcBuilding", name="Building"
        )
        storey = ifcopenshell.api.run(
            "root.create_entity",
            model,
            ifc_class="IfcBuildingStorey",
            name=storey_name,
        )

        ifcopenshell.api.run(
            "aggregate.assign_object", model, relating_object=project, products=[site]
        )
        ifcopenshell.api.run(
            "aggregate.assign_object", model, relating_object=site, products=[building]
        )
        ifcopenshell.api.run(
            "aggregate.assign_object",
            model,
            relating_object=building,
            products=[storey],
        )

        # ------------------------------------------------------------------
        # Create walls
        # ------------------------------------------------------------------
        wall_count = 0
        for i, ws in enumerate(floor_plan.walls):
            try:
                self._create_ifc_wall(
                    model,
                    body_context,
                    storey,
                    ws,
                    ceiling_height,
                    name=f"Wall_{i + 1}",
                )
                wall_count += 1
            except Exception as exc:
                logger.warning(f"Skipping wall {i + 1}: {exc}")

        logger.info(f"Created {wall_count} IFC walls.")

        # ------------------------------------------------------------------
        # Write file
        # ------------------------------------------------------------------
        model.write(output_path)
        abs_path = os.path.abspath(output_path)
        logger.info(f"IFC file written: {abs_path}")
        return abs_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_ifc_wall(
        self,
        model: ifcopenshell.file,
        body_context,
        storey,
        ws: WallSegment,
        ceiling_height: float,
        name: str,
    ) -> None:
        """Create a single IfcWall entity and assign it to the storey."""

        # Create geometry
        extrusion, placement_3d = _create_wall_geometry(
            model,
            body_context,
            ws.x1,
            ws.y1,
            ws.x2,
            ws.y2,
            ws.thickness,
            ceiling_height,
        )

        # Shape representation
        shape_repr = model.createIfcShapeRepresentation(
            body_context, "Body", "SweptSolid", [extrusion]
        )
        product_repr = model.createIfcProductDefinitionShape(None, None, [shape_repr])

        # Local placement
        world_origin = model.createIfcCartesianPoint([0.0, 0.0, 0.0])
        world_z = model.createIfcDirection([0.0, 0.0, 1.0])
        world_x = model.createIfcDirection([1.0, 0.0, 0.0])
        world_placement = model.createIfcAxis2Placement3D(world_origin, world_z, world_x)
        relative_placement = model.createIfcLocalPlacement(None, world_placement)
        local_placement = model.createIfcLocalPlacement(relative_placement, placement_3d)

        # Create the wall entity
        wall = ifcopenshell.api.run(
            "root.create_entity", model, ifc_class="IfcWall", name=name
        )
        wall.ObjectPlacement = local_placement
        wall.Representation = product_repr

        # Assign wall type attribute
        if hasattr(wall, "PredefinedType"):
            try:
                wall.PredefinedType = "STANDARD"
            except Exception:
                pass

        # Assign to storey
        ifcopenshell.api.run(
            "spatial.assign_container",
            model,
            relating_structure=storey,
            products=[wall],
        )
