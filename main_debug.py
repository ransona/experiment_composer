from core.timeline import Timeline
from core.canvas_composer import CanvasComposer
from core.writer import VideoWriter
from sources.video_bin_source import VideoBinSource
from sources.stimulus_video_source import StimulusVideoSource
from sources.line_plot_source import LinePlotSource
import os
import numpy as np


def main():
    # ------------------------------
    # --- Experiment configuration
    # ------------------------------
    # Example timeline: from 1000 s to 1020 s, 10 fps
    timeline = Timeline(1000, 1010, 10.0)

    exp_id = "2025-07-07_05_ESPM154"
    user = "pmateosaparicio"

    # Base directories for mapping Bonsai resource paths
    bonsai_root = "D:\\bonsai_resources\\"
    stim_base_dir = "/home/adamranson/data/vid_for_decoder/"

    # ------------------------------
    # --- Data sources
    # ------------------------------


    # for line plot 1
    time = np.linspace(0, 5000, 5000)
    traces = [np.sin(0.1*time), np.cos(0.1*time + 1)]
    config = {"user": "adam", "expID": "exp1"}

    traces = [np.sin(0.1*time), np.cos(0.1*time + 1)]
    plot_src1 = LinePlotSource(
        config={},
        time_vector=time,
        y_values=traces,
        colors=["cyan", "magenta"],
        title="Smoothed Line Plot",
        y_label="Signal (a.u.)",
        time_window=(-1, 1),
        y_range_mode="global",
        interpolate=True,  # NEW
    )
     

    sources = {
        "plot1": plot_src1
    }

    # ------------------------------
    # --- Canvas layout
    # ------------------------------
    layout_cfg = {
        "canvas_size": (200, 200),  # (width, height)
        "elements": {
            # Line plot visualizer (bottom, spanning full width)
            "line_plot": {
                "source": "plot1",  # name of your LinePlotSource instance
                "x": 0,
                "y": 0,               # positioned below the videos
                "w": 200,              # span full width of both videos
                "h": 200,               # height of the plot area
            },
        },
    }

    # ------------------------------
    # --- Initialize composer
    # ------------------------------
    composer = CanvasComposer(sources, layout_cfg)
    composer.initialize()



    # Prepare output writer
    frame0 = composer.draw_composite(timeline.times[0])
    H, W = frame0.shape[:2]
    writer = VideoWriter("canvas_output.mp4", fps=timeline.fps, frame_size=(W, H))

    # ------------------------------
    # --- Render loop
    # ------------------------------
    for t in timeline:
        frame = composer.draw_composite(t)
        writer.write(frame)

    writer.close()
    print("✅ Done — combined canvas video saved as canvas_output.mp4")


if __name__ == "__main__":
    main()
