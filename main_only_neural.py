from core.timeline import Timeline
from core.canvas_composer import CanvasComposer
from core.writer import VideoWriter
from sources.video_bin_source import VideoBinSource
from sources.stimulus_video_source import StimulusVideoSource
from sources.line_plot_source import LinePlotSource
import os
import numpy as np
import organise_paths
import pickle


def main():
    # ------------------------------
    # --- Experiment configuration
    # ------------------------------
    # Example timeline: from 1000 s to 1020 s, 10 fps
    timeline = Timeline(1945, 1990, 10.0)

    expID = "2025-07-04_06_ESPM154"
    userID = "pmateosaparicio"

    animalID, remote_repository_root, processed_root, exp_dir_processed, exp_dir_raw = organise_paths.find_paths(userID, expID)
    exp_dir_processed_recordings = os.path.join(exp_dir_processed,'recordings')
    exp_dir_processed_cut = os.path.join(exp_dir_processed,'cut')    
    Ch = 0

    # Base directories for mapping Bonsai resource paths
    bonsai_root = "D:\\bonsai_resources\\"
    stim_base_dir = "/home/adamranson/data/vid_for_decoder/"

    # ------------------------------
    # --- Data sources
    # ------------------------------

    # Neural imaging data
    video_cfg = {
        "user": "pmateosaparicio",
        "expID": "2025-07-07_05_ESPM154",
        "planes": [2,3,4,5],
        "height": 512,
        "width": 512,
        "spatial_sigma": 1.0,
        "temporal_window": 3,
        "enable_spatial_filter": False,
        "enable_temporal_filter": True,
        "interpolate": True,
        "tile_layout": {
            "rows": 2,
            "cols": 2,
            "order": [0,1,2,3],
            "gap": 4
        },
    }
    video_src = VideoBinSource(video_cfg)

    # for the stimulus video source
    stim_cfg = {
        "user": userID,
        "expID": expID,
    }
    stim_src = StimulusVideoSource(
        config=stim_cfg,
        bonsai_root=bonsai_root,
        stimulus_base_dir=stim_base_dir,
        fps=30,
    )

    # for line plot 1 - load OASIS neural activity 
    with open(os.path.join(exp_dir_processed_recordings,('s2p_oasis_ch' + str(Ch)+'.pickle')),'rb') as file: oasis_data = pickle.load(file)
    oasis_time = oasis_data['t']
    oasis_traces = np.mean(oasis_data['oasis_spikes'],axis=0).T

    plot_src1 = LinePlotSource(
        config={},
        time_vector=oasis_time,
        y_values=[oasis_traces],
        colors=["cyan", "magenta"],
        title="Population activity (spikes)",
        y_label="Signal (a.u.)",
        time_window=(-5, 5),
        y_range_mode="global",
        interpolate=True,  # NEW
    )
     

    sources = {
        "video0": video_src
    }

    # ------------------------------
    # --- Canvas layout
    # ------------------------------
    layout_cfg = {
        "canvas_size": (1008, 1008),  # (width, height)
        "elements": {
            # Neural activity video (left)
            "main_video": {
                "source": "video0",
                "x": 0,
                "y": 0,
                "w": 1000,
                "h": 1000,
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
