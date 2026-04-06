"""
BIM Agent MVP — CLI entry point.

Usage:
  python main.py demo
  python main.py process --input <path> [--output <dir>] [--scale N] [--ceiling-height H] [--show-viz]
  python main.py download [--output <dir>]
  python main.py visualize --input <path> [--output <dir>] [--show-viz]
"""

import argparse
import logging
import os
import sys

# Ensure project root is on sys.path regardless of invocation directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_demo(args: argparse.Namespace) -> None:
    """Generate a synthetic floor plan, parse it, export IFC, show viz."""
    from data.loader import DatasetLoader
    from agents.orchestrator import BIMOrchestrator

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== BIM Agent Demo ===\n")

    # 1. Generate synthetic floor plan
    print("[1/5] Generating synthetic L-shaped floor plan...")
    loader = DatasetLoader()
    synthetic_path = os.path.join(output_dir, "synthetic_floor_plan.png")
    loader.generate_synthetic(synthetic_path)
    print(f"      Saved: {synthetic_path}")

    # 2–5. Run full pipeline
    print("[2/5] Running BIM processing pipeline...")
    orchestrator = BIMOrchestrator()
    result = orchestrator.process(
        image_path=synthetic_path,
        output_dir=output_dir,
        pixels_per_meter=args.scale,
        ceiling_height=args.ceiling_height,
        show_viz=args.show_viz,
    )

    # Print summary
    print("\n=== Results ===")
    print(f"  IFC file:         {result['ifc_path']}")
    print(f"  2D visualization: {result['viz_2d_path']}")
    print(f"  3D visualization: {result['viz_3d_path']}")
    print(f"  Processing time:  {result['elapsed_s']} s")
    print("\n=== Floor Plan Statistics ===")
    for key, val in result["stats"].items():
        label = key.replace("_", " ").capitalize()
        print(f"  {label}: {val}")

    # Also print the floor plan text summary
    print("\n" + result["floor_plan"].summary())


def cmd_process(args: argparse.Namespace) -> None:
    """Process a given floor plan image."""
    if not args.input:
        print("ERROR: --input is required for the 'process' command.")
        sys.exit(1)

    from agents.orchestrator import BIMOrchestrator

    orchestrator = BIMOrchestrator()
    result = orchestrator.process(
        image_path=args.input,
        output_dir=args.output,
        pixels_per_meter=args.scale,
        ceiling_height=args.ceiling_height,
        show_viz=args.show_viz,
    )

    print("\n=== Processing Complete ===")
    print(f"  IFC file:         {result['ifc_path']}")
    print(f"  2D visualization: {result['viz_2d_path']}")
    print(f"  3D visualization: {result['viz_3d_path']}")
    print(f"  Processing time:  {result['elapsed_s']} s")
    for key, val in result["stats"].items():
        label = key.replace("_", " ").capitalize()
        print(f"  {label}: {val}")


def cmd_agent(args: argparse.Namespace) -> None:
    """Process a floor plan with the Claude-powered agentic pipeline."""
    if not args.input:
        print("ERROR: --input is required for the 'agent' command.")
        sys.exit(1)

    from agents.bim_agent import BIMAgent

    print("\n=== BIM Agent (Agentic Mode) ===\n")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Scale:  {args.scale} px/m")
    print(f"  Ceiling height: {args.ceiling_height} m")
    print("\nClaude will analyse the image and select algorithms autonomously.\n")

    agent = BIMAgent()
    result = agent.process(
        image_path=args.input,
        output_dir=args.output,
        pixels_per_meter=args.scale,
        ceiling_height=args.ceiling_height,
    )

    print("\n=== Results ===")
    print(f"  IFC file:         {result.get('ifc_path')}")
    print(f"  2D visualization: {result.get('viz_2d_path')}")
    print(f"  3D visualization: {result.get('viz_3d_path')}")
    print(f"  Top-view 3D:      {result.get('viz_top_path')}")
    print(f"  Processing time:  {result.get('elapsed_s')} s")
    if result.get("stats"):
        print("\n=== Floor Plan Statistics ===")
        for key, val in result["stats"].items():
            label = key.replace("_", " ").capitalize()
            print(f"  {label}: {val}")
    if result.get("summary"):
        print(f"\n=== Agent Summary ===\n{result['summary']}")


def cmd_download(args: argparse.Namespace) -> None:
    """Download CubiCasa5k sample images."""
    from data.loader import DatasetLoader

    loader = DatasetLoader()
    data_dir = os.path.join(args.output, "cubicasa_samples")
    print(f"Downloading CubiCasa5k samples to {data_dir} ...")
    paths = loader.download_cubicasa_sample(data_dir)
    if paths:
        print(f"Downloaded {len(paths)} images:")
        for p in paths:
            print(f"  {p}")
    else:
        print("No images downloaded (check network connection).")


def cmd_visualize(args: argparse.Namespace) -> None:
    """Parse a floor plan and show/save its visualizations."""
    if not args.input:
        print("ERROR: --input is required for the 'visualize' command.")
        sys.exit(1)

    from agents.floor_plan_parser import FloorPlanParser
    from export.visualizer import Visualizer

    parser = FloorPlanParser()
    floor_plan = parser.parse(args.input, pixels_per_meter=args.scale)

    os.makedirs(args.output, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]

    viz = Visualizer()

    viz_2d = os.path.join(args.output, f"{base}_2d.png")
    saved_2d = viz.plot_floor_plan(floor_plan, output_path=viz_2d, show=args.show_viz)
    print(f"2D visualization: {saved_2d}")

    viz_3d = os.path.join(args.output, f"{base}_3d.png")
    saved_3d = viz.plot_3d_preview(
        floor_plan,
        ceiling_height=args.ceiling_height,
        output_path=viz_3d,
        show=args.show_viz,
    )
    print(f"3D visualization: {saved_3d}")

    print("\n" + floor_plan.summary())


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bim_agent",
        description="BIM Agent MVP: convert 2D floor plan images to 3D IFC BIM models.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    # Shared options factory
    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--output", default="./output",
            help="Output directory (default: ./output)"
        )
        p.add_argument(
            "--scale", type=float, default=100.0,
            metavar="PIXELS_PER_METER",
            help="Pixels per metre (default: 100)"
        )
        p.add_argument(
            "--ceiling-height", type=float, default=2.8,
            metavar="HEIGHT_M",
            help="Ceiling height in metres (default: 2.8)"
        )
        p.add_argument(
            "--show-viz", action="store_true",
            help="Display matplotlib visualization windows"
        )

    # agent
    p_agent = sub.add_parser("agent", help="Process a floor plan image → IFC using the Claude agent")
    p_agent.add_argument("--input", help="Path to floor plan image")
    add_common(p_agent)

    # demo
    p_demo = sub.add_parser("demo", help="Run demo with synthetic floor plan")
    add_common(p_demo)

    # process
    p_process = sub.add_parser("process", help="Process a floor plan image → IFC")
    p_process.add_argument("--input", help="Path to floor plan image")
    add_common(p_process)

    # download
    p_dl = sub.add_parser("download", help="Download sample CubiCasa5k data")
    add_common(p_dl)

    # visualize
    p_viz = sub.add_parser("visualize", help="Show visualization of a floor plan")
    p_viz.add_argument("--input", help="Path to floor plan image")
    add_common(p_viz)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(verbose=getattr(args, "verbose", False))

    dispatch = {
        "agent": cmd_agent,
        "demo": cmd_demo,
        "process": cmd_process,
        "download": cmd_download,
        "visualize": cmd_visualize,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
