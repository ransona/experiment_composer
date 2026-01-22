from core.timeline import Timeline
from core.canvas_composer import CanvasComposer
from core.writer import VideoWriter
from sources.video_bin_source import VideoBinSource
from sources.stimulus_video_source import StimulusVideoSource
from sources.line_plot_source import LinePlotSource
from sources.reconstruction_video_source import ReconstructionVideoSource 
from sources.eye_source import EyeSource
import os
import numpy as np
import organise_paths
import pickle
from tqdm.auto import tqdm
from datetime import datetime

## TMUX: to run in tmux start tmux with tmux new -s gui


def main():
    # ------------------------------
    # --- Experiment configuration
    # ------------------------------
    # Example timeline: from 1000 s to 1020 s, 10 fps
    timeline = Timeline(10, 900, 20.0)
    playback_multiplier = 3

    expID = "2025-07-04_06_ESPM154"
    userID = "pmateosaparicio"

    output_dir = '/home/adamranson/data/composer_output'

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
        "expID": expID,
        "planes": [2,4,6,8],
        "height": 512,
        "width": 512,
        "spatial_sigma": 1.0,
        "temporal_window": 6,
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

    recon_src = ReconstructionVideoSource(
        video_path=os.path.join(exp_dir_processed, "reconstruction", "session_recons_cut.mp4"),
        timestamps_path=os.path.join(exp_dir_processed, "reconstruction", "video_timeline.npy"),
        enable_temporal_filter=True,   # enable/disable
        temporal_window=3,             # frames on each side → 7-frame box
        enable_spatial_filter=True,    # enable/disable
        spatial_sigma=1.2,             # pixels (Gaussian sigma)
        interpolate=True,              # linear blend between adjacent frames
        cache_size=96,                 # small LRU cache for on-demand decoding
    )

    head_src = EyeSource(
        exp_dir_processed=exp_dir_processed,
        expID=expID,
        eye="right",                        # or "left"
        timestamps_path=os.path.join(exp_dir_processed, "recordings", "eye_frame_times.npy"),
        crop=False,                         # True → interactive ROI; or (x,y,w,h)
        plot_detected_pupil=True,
        plot_detected_eye=True,
        overlay_thickness=2,
        contrast_clip_percentiles=(0, 90.0),
    )  



    # for line plot 1 - load OASIS neural activity 
    with open(os.path.join(exp_dir_processed_recordings,('s2p_oasis_ch' + str(Ch)+'.pickle')),'rb') as file: oasis_data = pickle.load(file)
    oasis_time = oasis_data['t']
    oasis_traces = np.mean(oasis_data['oasis_spikes'],axis=0).T

    plot_oasis = LinePlotSource(
        config={},
        time_vector=oasis_time,
        y_values=[oasis_traces],
        colors=["cyan", "magenta"],
        title="Mean population activity",
        y_label="",
        time_window=(-5, 0),
        y_range_mode="global",
        interpolate=True,  # NEW
    )
     

    # for line plot 2 - running speed
    with open(os.path.join(exp_dir_processed_recordings,('s2p_oasis_ch' + str(Ch)+'.pickle')),'rb') as file: oasis_data = pickle.load(file)

    wheel_data = pickle.load(open(os.path.join(exp_dir_processed_recordings,('wheel.pickle')), "rb"))

    wheel_time = wheel_data['t']
    wheel_trace = wheel_data['speed']

    plot_wheel = LinePlotSource(
        config={},
        time_vector=wheel_time,
        y_values=[wheel_trace],
        colors=["cyan"],
        title="Run speed",
        y_label="",
        time_window=(-5, 5),
        y_range_mode="global",
        interpolate=True,  # NEW
    )

    sources = {
        "video0": video_src,
        "stimulus": stim_src,
        "reconstruction": recon_src,
        #"reconstruction_stim": recon_stim_src,
        "plot_oasis": plot_oasis,
        "plot_wheel": plot_wheel,    
        "head": head_src
    }

    # ------------------------------
    # --- Canvas layout
    # ------------------------------
    layout_cfg = {
        # widened to fit a third 500px panel in the top row
        "canvas_size": (800, 1500),  # (height,width)
        "elements": {
            # Neural activity video (left)
            "main_video": {
                "source": "video0",
                "x": 0,
                "y": 0,
                "w": 500,
                "h": 500,
            },

            # # Stimulus presentation video (middle)
            # "reconstruction_stimulus_video": {
            #     "source": "reconstruction_stim",
            #     "x": 500,
            #     "y": 0,
            #     "w": 250,
            #     "h": 250,
            # },

            # Reconstruction video 
            "reconstruction_video": {
                "source": "reconstruction",
                "x": 750,
                "y": 0,
                "w": 250,
                "h": 250,
            },

            # # stim video 
            # "stim_video": {
            #     "source": "stimulus",
            #     "x": 500,
            #     "y": 250,
            #     "w": 250,
            #     "h": 250,
            # },            

            # Head
            "head": {
                "source": "head",
                "x": 750,
                "y": 250,
                "w": 250,
                "h": 250,
            },                   

            # OASIS Line plot visualizer (bottom, left)
            "line_plot_oasis": {
                "source": "plot_oasis",  # name of your LinePlotSource instance
                "x": 0,
                "y": 500,               # positioned below the videos
                "w": 500,               # width of the plot area
                "h": 300,               # height of the plot area
            },

            # WHEEL Line plot visualizer (bottom, middle)
            "line_plot_wheel": {
                "source": "plot_wheel",  # name of your LinePlotSource instance
                "x": 500,
                "y": 500,               # positioned below the videos
                "w": 500,               # width of the plot area
                "h": 300,               # height of the plot area
            }
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
    now = datetime.now()
    filename = now.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
    writer = VideoWriter(os.path.join(output_dir,filename), fps=timeline.fps*playback_multiplier, frame_size=(W, H))

    # ------------------------------
    # --- Render loop
    # ------------------------------
    for t in tqdm(timeline.times, desc="Rendering", unit="frame"):
        frame = composer.draw_composite(t)
        writer.write(frame)

    writer.close()
    print("✅ Done — combined canvas video saved as " + filename)


if __name__ == "__main__":
    main()
