import os
import json
import pathlib
import logging

import cv2
import numpy as np

import pandas as pd

from tqdm import tqdm
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def analyze_frame_brightness(grayscale, threshold=None):
    """
    Analyze brightness of a single grayscale frame using Otsu's method.
    Handles edge cases like uniform brightness.
    """
    # Apply Otsu's thresholding
    if threshold is None:
        threshold, _ = cv2.threshold(
            grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    mask = cv2.threshold(grayscale, threshold, 255, cv2.THRESH_BINARY)[1]

    # Compute brightness
    background_mask = mask == 0
    object_mask = mask == 255
    bg_brightness = (
        np.mean(grayscale[background_mask]) if np.any(background_mask) else 0
    )
    obj_brightness = np.mean(grayscale[object_mask]) if np.any(object_mask) else 0
    return float(bg_brightness), float(obj_brightness)


def compute_brightness_data(video_path, output_dir: str | pathlib.Path):
    """
    Analyzes brightness of a bright object and dark background in each video frame.
    """
    if not isinstance(output_dir, pathlib.Path):
        output_dir = pathlib.Path(output_dir)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    brightness_data = []

    # Open video
    logger.info(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    brightness_data = {
        "frame": [],
        "time": [],
        "background": [],
        "object": [],
        "difference": [],
    }

    if logger.isEnabledFor(logging.INFO):
        pbar = tqdm(
            total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            desc="Processing video",
            unit="frame",
        )
    else:
        pbar = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bg, obj = analyze_frame_brightness(grayscale)
        brightness_data["frame"].append(frame_count)
        brightness_data["time"].append(frame_count / fps)
        brightness_data["background"].append(bg)
        brightness_data["object"].append(obj)
        brightness_data["difference"].append(obj - bg)
        frame_count += 1
        if pbar is not None:
            pbar.update(1)
    cap.release()

    if pbar is not None:
        pbar.close()

    logger.info("Video processing complete.")
    logger.info("Saving brightness data...")

    # Save data
    df = pd.DataFrame(brightness_data)
    df.to_excel(output_dir / "brightness_data.xlsx", index=False)
    with open(output_dir / "brightness_data.json", "w", encoding="utf-8") as f:
        json.dump(brightness_data, f, indent=2)
    logger.info("Brightness data saved successfully.")
    return brightness_data


def generate_plots(data, plot_config, output_dir: str | pathlib.Path):
    """Generate and save brightness plots using Plotly."""
    logger.info("Generating brightness over time plots...")

    if not isinstance(output_dir, pathlib.Path):
        output_dir = pathlib.Path(output_dir)

    os.makedirs(output_dir / "plots", exist_ok=True)

    times = data["time"]
    bg_list = data["background"]
    obj_list = data["object"]
    diff_list = data["difference"]

    defaults = plot_config["defaults"]

    def save_plot(y_label, values, filename, params):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=values, **params["points"]))
        fig.update_layout(
            title=params["title"], xaxis_title="Time (s)", yaxis_title=y_label
        )
        fig.write_image(
            f"{filename}.png",
            width=defaults["width"],
            height=defaults["height"],
            scale=defaults["scale"],
        )

    save_plot(
        "Brightness",
        bg_list,
        output_dir / "plots" / "background",
        plot_config["background"],
    )
    save_plot(
        "Brightness",
        obj_list,
        output_dir / "plots" / "object",
        plot_config["object"],
    )
    save_plot(
        "Difference",
        diff_list,
        output_dir / "plots" / "difference",
        plot_config["difference"],
    )

    logger.info("Brightness over time plots generated successfully.")
