"""Dataset loader: synthetic floor plan generator and CubiCasa5k downloader."""

import logging
import os
import warnings
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Utilities for obtaining floor plan images for testing."""

    # ------------------------------------------------------------------
    # Synthetic floor plan
    # ------------------------------------------------------------------

    def generate_synthetic(self, output_path: str) -> str:
        """
        Draw a synthetic L-shaped apartment floor plan and save as PNG.

        Layout (800×600 px, 1 px = 1 cm → ~8 m × 6 m):
        - Exterior L-shape: full rectangle minus top-right quadrant
        - 4 rooms: Living Room (bottom-left), Bedroom (top-left),
                   Kitchen (bottom-right), Bathroom (top-right portion)
        - Interior partition walls
        - Door gaps and window markers

        Args:
            output_path: Where to save the generated PNG.

        Returns:
            Absolute path to the saved PNG.
        """
        W, H = 800, 600
        WALL = 10       # wall thickness in pixels
        BG = (255, 255, 255)
        WALL_COLOR = (0, 0, 0)
        DOOR_COLOR = (139, 90, 43)   # brown
        WIN_COLOR = (70, 130, 180)   # steel blue

        img = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(img)

        # ------------------------------------------------------------------
        # L-shape exterior walls
        # The L-shape: full 800×600 minus the top-right quadrant (400×300)
        #   i.e. the building covers:
        #     bottom-left rectangle: (0,300) → (800,600)   [full width, bottom half]
        #     top-left rectangle:    (0,0)   → (400,300)   [left half, top half]
        # ------------------------------------------------------------------

        def rect(x0, y0, x1, y1, thickness=WALL, color=WALL_COLOR):
            """Draw a hollow rectangle (just the border walls)."""
            draw.rectangle([x0, y0, x1, y0 + thickness], fill=color)  # top
            draw.rectangle([x0, y1 - thickness, x1, y1], fill=color)  # bottom
            draw.rectangle([x0, y0, x0 + thickness, y1], fill=color)  # left
            draw.rectangle([x1 - thickness, y0, x1, y1], fill=color)  # right

        def hline(x0, y, x1, thickness=WALL, color=WALL_COLOR):
            draw.rectangle([x0, y, x1, y + thickness], fill=color)

        def vline(x, y0, y1, thickness=WALL, color=WALL_COLOR):
            draw.rectangle([x, y0, x + thickness, y1], fill=color)

        # Exterior: bottom full-width rectangle
        rect(0, 300, W, H)
        # Exterior: top-left rectangle (shares bottom edge with bottom rect)
        rect(0, 0, W // 2, 300 + WALL)  # +WALL to share wall with bottom

        # The L-notch (top-right) boundary:
        # Horizontal wall from mid-x to right edge at y=300
        hline(W // 2 - WALL, 300, W)
        # Vertical wall from top to y=300 at x=400
        # (already drawn by right side of top-left rect)

        # ------------------------------------------------------------------
        # Interior partition walls
        # ------------------------------------------------------------------

        # Vertical wall splitting Living Room / Kitchen (bottom, x=400)
        # Has a door gap at y≈480..560
        vline(W // 2, 300, 480)
        vline(W // 2, 560, H)

        # Horizontal wall splitting Bedroom / Bathroom (top-left, y=150)
        # Has a door gap at x≈80..160
        hline(80, 150, W // 2 - WALL)
        hline(0, 150, 80)               # left portion before door
        # (gap from x=80 to x=160 is the door)
        hline(160, 150, W // 2 - WALL)

        # ------------------------------------------------------------------
        # Doors (brown rectangles = gaps / swing indicators)
        # ------------------------------------------------------------------

        # Front door (exterior, bottom wall, centre)
        draw.rectangle([W // 2 - 45, H - WALL - 2, W // 2 + 45, H + 2], fill=BG)
        draw.rectangle([W // 2 - 45, H - WALL, W // 2 + 45, H], fill=DOOR_COLOR)

        # Interior door: Living ↔ Kitchen (vertical wall gap at y=480..560)
        draw.rectangle([W // 2 - 2, 480, W // 2 + WALL + 2, 560], fill=BG)
        draw.rectangle([W // 2, 480, W // 2 + WALL, 560], fill=DOOR_COLOR)

        # Interior door: Bedroom ↔ Bathroom (horizontal wall gap at x=80..160)
        draw.rectangle([80, 150 - 2, 160, 150 + WALL + 2], fill=BG)
        draw.rectangle([80, 150, 160, 150 + WALL], fill=DOOR_COLOR)

        # Bedroom exterior door (left wall)
        draw.rectangle([0, 60, WALL + 2, 140], fill=BG)
        draw.rectangle([0, 60, WALL, 140], fill=DOOR_COLOR)

        # ------------------------------------------------------------------
        # Windows (blue dashes on exterior walls)
        # ------------------------------------------------------------------

        # Living room: bottom wall windows
        for wx in [120, 260]:
            draw.rectangle([wx - 30, H - WALL, wx + 30, H], fill=WIN_COLOR)

        # Kitchen: right wall window
        draw.rectangle([W - WALL, 400, W, 500], fill=WIN_COLOR)

        # Bedroom: top wall window
        draw.rectangle([80, 0, 200, WALL], fill=WIN_COLOR)

        # Bathroom: left portion top wall window
        draw.rectangle([280, 0, 360, WALL], fill=WIN_COLOR)

        # ------------------------------------------------------------------
        # Save
        # ------------------------------------------------------------------
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        img.save(output_path)
        logger.info(f"Synthetic floor plan saved: {output_path}")
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # CubiCasa5k sample downloader
    # ------------------------------------------------------------------

    def download_cubicasa_sample(self, output_dir: str) -> List[str]:
        """
        Download CubiCasa5k sample floor plan images from the official notebook.

        The full dataset (5.4 GB) is on Zenodo. This method extracts the
        embedded sample PNGs from the CubiCasa5k samples.ipynb notebook on
        GitHub — no account or large download required.

        Args:
            output_dir: Directory to save downloaded images.

        Returns:
            List of paths to successfully downloaded images.
        """
        import base64
        import requests

        os.makedirs(output_dir, exist_ok=True)

        notebook_url = (
            "https://raw.githubusercontent.com/CubiCasa/CubiCasa5k/"
            "master/samples.ipynb"
        )
        downloaded: List[str] = []

        try:
            logger.info(f"Fetching CubiCasa5k samples notebook from GitHub...")
            resp = requests.get(notebook_url, timeout=60)
            resp.raise_for_status()
            notebook = resp.json()
        except Exception as exc:
            warnings.warn(f"Could not fetch CubiCasa5k notebook: {exc}", stacklevel=2)
            return downloaded

        idx = 0
        for cell in notebook.get("cells", []):
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                if "image/png" not in data:
                    continue
                img_bytes = base64.b64decode(data["image/png"])
                # Only keep large images — floor plan renders, not tiny charts
                if len(img_bytes) < 50_000:
                    continue
                dest = os.path.join(output_dir, f"cubicasa_sample_{idx}.png")
                if os.path.exists(dest):
                    logger.info(f"Already exists: {dest}")
                else:
                    with open(dest, "wb") as fh:
                        fh.write(img_bytes)
                    logger.info(f"Saved: {dest} ({len(img_bytes) // 1024} KB)")
                downloaded.append(dest)
                idx += 1

        logger.info(f"Extracted {len(downloaded)} CubiCasa sample images.")
        return downloaded

    # ------------------------------------------------------------------
    # List available samples
    # ------------------------------------------------------------------

    def list_samples(self, data_dir: str) -> List[str]:
        """
        List all floor plan images in a directory.

        Recognises .png, .jpg, .jpeg, .bmp, .tiff extensions.

        Args:
            data_dir: Directory to scan.

        Returns:
            Sorted list of absolute image file paths.
        """
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
        data_path = Path(data_dir)
        if not data_path.is_dir():
            logger.warning(f"Directory not found: {data_dir}")
            return []

        images = sorted(
            str(p.resolve())
            for p in data_path.iterdir()
            if p.suffix.lower() in extensions
        )
        logger.info(f"Found {len(images)} images in {data_dir}")
        return images
