from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip  # type: ignore
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def _gather_pot_frames_from_csv(
    csv_path: Path, plant_id: Optional[int] = None
) -> Dict[int, List[Path]]:
    """Gather pot image frames from all.csv, grouped by plant_id.

    Args:
        csv_path: Path to all.csv file
        plant_id: Optional specific plant_id to filter for

    Returns:
        Dictionary mapping plant_id to sorted list of image paths
    """
    df = pd.read_csv(csv_path)

    # Filter by plant_id if specified
    if plant_id is not None:
        df = df[df["plant_id"] == plant_id]

    # Group by plant_id and gather paths
    frames_by_plant = {}
    for pid in df["plant_id"].unique():
        plant_df = df[df["plant_id"] == pid].sort_values("image_name")
        # Convert relative paths to absolute based on CSV location
        csv_dir = csv_path.parent
        frames = [csv_dir / path for path in plant_df["pot_image_path"]]
        # Filter to only existing files
        frames = [p for p in frames if p.exists()]
        if frames:
            frames_by_plant[int(pid)] = frames

    return frames_by_plant


def _extract_timestamp_and_day(
    filename: str, first_date: datetime | None = None
) -> tuple[str, int | None]:
    """Extract timestamp and calculate day number from filename.

    Filename format: 2025-10-08T153000+0000_left.jpg (parent dir has timestamp)
    Returns: (formatted_timestamp, day_number)
    """
    try:
        # Handle both direct timestamp filenames and parent directory timestamps
        # Try extracting from filename first
        parts = filename.split("_")
        if len(parts) > 0 and "T" in parts[0]:
            timestamp_str = parts[0]
        else:
            # Fallback: try to parse the whole filename
            timestamp_str = filename.split(".")[0]

        # Parse ISO format: 2025-10-08T153000+0000
        dt = datetime.strptime(
            timestamp_str.split(".")[0].split("_")[0], "%Y-%m-%dT%H%M%S%z"
        )

        # Format as human-readable
        formatted = dt.strftime("%Y-%m-%d %H:%M:%S")

        # Calculate day number if we have a reference date
        day_num = None
        if first_date is not None:
            delta = (dt.date() - first_date.date()).days
            day_num = delta

        return formatted, day_num
    except Exception as e:
        print(f"Warning: Could not parse timestamp from {filename}: {e}")
        return filename, None


def _add_text_overlay(
    image: Image.Image, timestamp: str, day_num: int | None
) -> Image.Image:
    """Add timestamp and day number text overlay to image."""
    # Create a copy to avoid modifying original
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Try to use a nice font, fall back to default if not available
    try:
        font_size = max(20, img.height // 30)
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except Exception:
        font = ImageFont.load_default()

    # Prepare text
    lines = [timestamp]
    if day_num is not None:
        lines.append(f"Day {day_num}")

    # Calculate text dimensions and position (top-left with padding)
    padding = max(10, img.height // 100)
    y_offset = padding

    for line in lines:
        # Get text bounding box
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Draw background rectangle for better readability
        bg_padding = 5
        draw.rectangle(
            [
                padding - bg_padding,
                y_offset - bg_padding,
                padding + text_width + bg_padding,
                y_offset + text_height + bg_padding,
            ],
            fill=(0, 0, 0, 180),
        )

        # Draw text in white
        draw.text((padding, y_offset), line, font=font, fill=(255, 255, 255))

        y_offset += text_height + bg_padding * 2

    return img


def _calculate_grid_layout(num_plants: int) -> Tuple[int, int]:
    """Calculate optimal grid layout (rows, cols) for given number of plants.

    Args:
        num_plants: Number of plants to display

    Returns:
        Tuple of (rows, cols)
    """
    cols = math.ceil(math.sqrt(num_plants))
    rows = math.ceil(num_plants / cols)
    return rows, cols


def _create_composite_frame(
    plant_frames: Dict[int, Path],
    plant_ids: List[int],
    target_size: Tuple[int, int],
    timestamp: str,
    day_num: Optional[int],
    rows: Optional[int] = None,
    cols: Optional[int] = None,
) -> np.ndarray:
    """Create a composite grid frame from multiple plant images.

    Args:
        plant_frames: Dictionary mapping plant_id to image path for this frame
        plant_ids: Sorted list of plant IDs to include
        target_size: Target size for each plant cell (width, height)
        timestamp: Timestamp string for overlay
        day_num: Day number for overlay
        rows: Optional number of rows (if None, calculated automatically)
        cols: Optional number of columns (if None, calculated automatically)

    Returns:
        Composite frame as numpy array
    """
    if rows is None or cols is None:
        rows, cols = _calculate_grid_layout(len(plant_ids))
    cell_w, cell_h = target_size

    # Create blank canvas
    canvas_w = cols * cell_w
    canvas_h = rows * cell_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(0, 0, 0))

    # Place each plant image in grid
    for idx, plant_id in enumerate(plant_ids):
        row = idx // cols
        col = idx % cols
        x = col * cell_w
        y = row * cell_h

        if plant_id in plant_frames and plant_frames[plant_id].exists():
            try:
                with Image.open(plant_frames[plant_id]) as img:
                    img = img.convert("RGB")
                    # Resize to fit cell while maintaining aspect ratio
                    img.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
                    # Center in cell
                    paste_x = x + (cell_w - img.width) // 2
                    paste_y = y + (cell_h - img.height) // 2
                    canvas.paste(img, (paste_x, paste_y))

                    # Add plant ID label
                    draw = ImageDraw.Draw(canvas)
                    try:
                        font = ImageFont.truetype(
                            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
                        )
                    except Exception:
                        font = ImageFont.load_default()

                    label = f"Plant {plant_id}"
                    bbox = draw.textbbox((0, 0), label, font=font)
                    label_w = bbox[2] - bbox[0]
                    label_h = bbox[3] - bbox[1]
                    label_x = x + 5
                    label_y = y + 5

                    # Draw label background
                    draw.rectangle(
                        [
                            label_x - 3,
                            label_y - 3,
                            label_x + label_w + 3,
                            label_y + label_h + 3,
                        ],
                        fill=(0, 0, 0, 200),
                    )
                    draw.text(
                        (label_x, label_y), label, font=font, fill=(255, 255, 255)
                    )
            except Exception as e:
                print(f"  Warning: Could not load plant {plant_id}: {e}")

    # Add timestamp overlay to entire composite
    canvas = _add_text_overlay(canvas, timestamp, day_num)

    return np.array(canvas)


def _build_video(frames: List[Path], out_path: Path, fps: int = 2) -> None:
    """Build a video from a list of image frames."""
    if len(frames) == 0:
        print(f"No frames found for {out_path.name}; skipping.")
        return
    if len(frames) == 1:
        print(f"Only 1 frame for {out_path.name}; duplicating to create a short video.")
        frames = frames * 2

    # Determine target canvas size across all frames
    sizes = []
    for p in frames:
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
        except Exception:
            pass
    if not sizes:
        print(f"No readable frames for {out_path.name}; skipping.")
        return
    max_w = max(w for (w, h) in sizes)
    max_h = max(h for (w, h) in sizes)

    # Determine the first date for day numbering from parent directory name
    first_date = None
    if frames:
        # Parent directory name should contain the timestamp
        parent_name = frames[0].parent.name
        try:
            first_date = datetime.strptime(
                parent_name.split("_")[0], "%Y-%m-%dT%H%M%S%z"
            )
        except Exception:
            # Fallback: try to extract from path
            try:
                timestamp_str = parent_name
                first_date = datetime.strptime(timestamp_str, "%Y-%m-%dT%H%M%S%z")
            except Exception:
                pass

    # Load, pad to common size, add text overlay, and convert to RGB numpy arrays
    arr_frames = []
    for p in frames:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                w, h = im.size
                if (w, h) != (max_w, max_h):
                    canvas = Image.new("RGB", (max_w, max_h), color=(0, 0, 0))
                    # center paste
                    ox = (max_w - w) // 2
                    oy = (max_h - h) // 2
                    canvas.paste(im, (ox, oy))
                    im = canvas

                # Add timestamp and day number overlay using parent directory name
                parent_name = p.parent.name
                timestamp, day_num = _extract_timestamp_and_day(parent_name, first_date)
                im = _add_text_overlay(im, timestamp, day_num)

                arr_frames.append(np.array(im))
        except Exception as e:
            print(f"Skipping unreadable frame {p.name}: {e}")

    if not arr_frames:
        print(f"No valid frames for {out_path.name}; skipping.")
        return

    clip = ImageSequenceClip(arr_frames, fps=fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clip.write_videofile(
        str(out_path),
        codec="libx264",
        audio=False,
        fps=fps,
        preset="medium",
        bitrate="2000k",
        threads=0,
        logger=None,  # Suppress moviepy progress bar
    )
    clip.close()


def build_composite_timelapse(
    csv_path: Path,
    output_path: Path,
    fps: int = 2,
    plant_ids: Optional[List[int]] = None,
    cell_size: int = 400,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    subsample: int = 1,
) -> None:
    """Build a composite timelapse video showing all plants in a grid.

    Uses streaming to reduce memory usage by loading only one frame at a time.

    Args:
        csv_path: Path to all.csv file
        output_path: Output video file path
        fps: Frames per second
        plant_ids: Optional list of specific plant IDs to process
        cell_size: Size of each plant cell in the grid
        rows: Number of rows in grid (default: 3)
        cols: Number of columns in grid (default: 6)
        subsample: Take every nth frame (default: 1, meaning all frames)
    """
    df = pd.read_csv(csv_path)

    # Get all plant IDs and filter if specified
    all_plant_ids = sorted(df["plant_id"].unique())
    if plant_ids is not None:
        all_plant_ids = [pid for pid in all_plant_ids if pid in plant_ids]

    if not all_plant_ids:
        print("No plants found")
        return

    print(f"Creating composite video for {len(all_plant_ids)} plants")
    print(f"Plant IDs: {all_plant_ids}")

    # Get all unique timestamps (sorted) and apply subsampling
    all_timestamps = sorted(df["image_name"].unique())
    if subsample > 1:
        all_timestamps = all_timestamps[::subsample]
        print(
            f"Processing {len(all_timestamps)} frames (subsampled every {subsample} frames)"
        )
    else:
        print(f"Processing {len(all_timestamps)} frames")

    # Calculate or use provided grid layout
    if rows is None or cols is None:
        # Default to 3 rows × 6 columns
        if rows is None:
            rows = 3
        if cols is None:
            cols = 6
    print(f"Grid layout: {rows} rows × {cols} columns")

    # Determine first date for day numbering
    first_date = None
    if all_timestamps:
        try:
            first_date = datetime.strptime(
                all_timestamps[0].split("_")[0], "%Y-%m-%dT%H%M%S%z"
            )
        except Exception:
            pass

    csv_dir = csv_path.parent

    print("\nGenerating video frames...")
    frames_list = []
    for img_name in tqdm(all_timestamps):
        # Get all plant images for this timestamp
        timestamp_df = df[df["image_name"] == img_name]

        plant_frames = {}
        for _, row in timestamp_df.iterrows():
            if row["plant_id"] in all_plant_ids:
                img_path = csv_dir / row["pot_image_path"]
                plant_frames[row["plant_id"]] = img_path

        # Extract timestamp info
        timestamp_str, day_num = _extract_timestamp_and_day(img_name, first_date)

        # Create composite frame
        frame = _create_composite_frame(
            plant_frames,
            all_plant_ids,
            (cell_size, cell_size),
            timestamp_str,
            day_num,
            rows,
            cols,
        )
        frames_list.append(frame)

    if not frames_list:
        print("No frames generated")
        return

    if len(frames_list) == 1:
        print("Only 1 frame; duplicating to create a short video.")
        frames_list = frames_list * 2

    clip = ImageSequenceClip(frames_list, fps=fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing video to {output_path}...")
    clip.write_videofile(
        str(output_path),
        # codec="libx264",
        # audio=False,
        fps=fps,
        # preset="medium",
        # bitrate="5000k",
        # threads=0,
        # logger=None,
    )
    clip.close()
    print(f"✓ Composite video saved to {output_path}")


def build_timelapse_videos_from_csv(
    csv_path: Path,
    output_dir: Path,
    fps: int = 2,
    plant_ids: Optional[List[int]] = None,
    composite: bool = False,
    cell_size: int = 400,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    subsample: int = 1,
) -> None:
    """Build timelapse videos for each plant from all.csv.

    Args:
        csv_path: Path to all.csv file
        output_dir: Output directory for videos
        fps: Frames per second
        plant_ids: Optional list of specific plant IDs to process
        composite: If True, create a composite video instead of individual videos
        cell_size: Size of each cell in composite grid
        rows: Number of rows in composite grid (default: 3)
        cols: Number of columns in composite grid (default: 6)
        subsample: Take every nth frame (default: 1, meaning all frames)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if composite:
        # Create composite video
        output_path = output_dir / "composite_timelapse.mp4"
        build_composite_timelapse(
            csv_path, output_path, fps, plant_ids, cell_size, rows, cols, subsample
        )
        return

    # Gather frames by plant
    frames_by_plant = _gather_pot_frames_from_csv(csv_path)

    if not frames_by_plant:
        print(f"No pot frames found in {csv_path}")
        return

    # Filter by plant_ids if specified
    if plant_ids is not None:
        frames_by_plant = {
            pid: frames for pid, frames in frames_by_plant.items() if pid in plant_ids
        }

    # Apply subsampling
    if subsample > 1:
        frames_by_plant = {
            pid: frames[::subsample] for pid, frames in frames_by_plant.items()
        }
        print(
            f"Found {len(frames_by_plant)} plants with images (subsampled every {subsample} frames)"
        )
    else:
        print(f"Found {len(frames_by_plant)} plants with images")

    # Build video for each plant
    for plant_id, frames in sorted(frames_by_plant.items()):
        print(f"\nBuilding timelapse for plant {plant_id} ({len(frames)} frames)...")
        out_path = output_dir / f"plant_{plant_id:03d}_timelapse.mp4"
        _build_video(frames, out_path, fps=fps)
        print(f"  ✓ Saved to {out_path}")


def build_timelapse_videos(
    fps: int = 2, boxes: bool = True, masks: bool = True
) -> None:
    """Legacy function for backward compatibility with test_images structure.

    This function is deprecated. Use build_timelapse_videos_from_csv instead.
    """
    print(
        "Warning: build_timelapse_videos is deprecated. Use build_timelapse_videos_from_csv instead."
    )
    print("This function expects the old test_images/grounding_dino structure.")


def find_processed_directories(base_path: Path, version: str = "v5.0.0") -> List[Path]:
    """Find all processed/version directories in the given base path.

    Args:
        base_path: Base path to search
        version: Version string to match

    Returns:
        List of paths to processed/version directories
    """
    processed_dirs = []

    # Search for processed/{version} directories
    for processed_dir in base_path.rglob(f"processed/{version}"):
        if processed_dir.is_dir():
            csv_path = processed_dir / "all.csv"
            if csv_path.exists():
                processed_dirs.append(processed_dir)

    return sorted(processed_dirs)


def main():
    parser = argparse.ArgumentParser(
        description="Create MP4 timelapses from processed pot images",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to all.csv file or processed directory containing all.csv",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory for videos (default: videos/ in the processed directory)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second for the output video",
    )
    parser.add_argument(
        "--plants",
        nargs="+",
        type=int,
        help="Specific plant IDs to process (default: all plants)",
    )
    parser.add_argument(
        "--composite",
        action="store_true",
        help="Create a composite video showing all plants in a grid (default: individual videos)",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=400,
        help="Size of each plant cell in composite grid (default: 400)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        help="Number of rows in composite grid (default: 3)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        help="Number of columns in composite grid (default: 6)",
    )
    parser.add_argument(
        "--scan",
        type=str,
        help="Scan for all processed directories under this path and create videos for each",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v5.0.0",
        help="Version string when using --scan (default: v5.0.0)",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=12,
        help="Take every nth frame",
    )
    args = parser.parse_args()

    if args.scan:
        # Scan mode: find all processed directories
        base_path = Path(args.scan)
        if not base_path.exists():
            print(f"Error: Scan path does not exist: {base_path}")
            return

        print(f"Scanning for processed/{args.version} directories in {base_path}...")
        processed_dirs = find_processed_directories(base_path, args.version)

        if not processed_dirs:
            print(f"No processed/{args.version} directories found")
            return

        print(f"Found {len(processed_dirs)} processed directories")

        for i, processed_dir in enumerate(processed_dirs, 1):
            print(f"\n{'=' * 80}")
            print(f"[{i}/{len(processed_dirs)}] Processing: {processed_dir}")
            print(f"{'=' * 80}")

            csv_path = processed_dir / "all.csv"
            output_dir = processed_dir / "videos"

            try:
                build_timelapse_videos_from_csv(
                    csv_path,
                    output_dir,
                    fps=args.fps,
                    plant_ids=args.plants,
                    composite=args.composite,
                    cell_size=args.cell_size,
                    rows=args.rows,
                    cols=args.cols,
                    subsample=args.subsample,
                )
            except Exception as e:
                print(f"Error processing {processed_dir}: {e}")
                import traceback

                traceback.print_exc()

        print(f"\n{'=' * 80}")
        print(f"Completed processing {len(processed_dirs)} directories")
        print(f"{'=' * 80}")

    else:
        # Single directory mode
        input_path = Path(args.input)

        if not input_path.exists():
            print(f"Error: Input path does not exist: {input_path}")
            return

        # Determine CSV path
        if input_path.is_file() and input_path.name == "all.csv":
            csv_path = input_path
            processed_dir = input_path.parent
        elif input_path.is_dir():
            csv_path = input_path / "all.csv"
            processed_dir = input_path
            if not csv_path.exists():
                print(f"Error: all.csv not found in {input_path}")
                return
        else:
            print("Error: Input must be all.csv file or directory containing all.csv")
            return

        # Determine output directory
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = processed_dir / "videos"

        print(f"Input CSV: {csv_path}")
        print(f"Output directory: {output_dir}")

        build_timelapse_videos_from_csv(
            csv_path,
            output_dir,
            fps=args.fps,
            plant_ids=args.plants,
            composite=args.composite,
            cell_size=args.cell_size,
            rows=args.rows,
            cols=args.cols,
            subsample=args.subsample,
        )


if __name__ == "__main__":
    main()
