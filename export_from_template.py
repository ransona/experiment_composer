import argparse
import json
import os

import organise_paths
from core.canvas_composer import CanvasComposer
from core.timeline import Timeline
from core.writer import VideoWriter
from sources.video_bin_source import VideoBinSource
from sources.stimulus_video_source import StimulusVideoSource
from sources.stimulus_source import StimulusSource
from sources.reconstruction_video_source import ReconstructionVideoSource
from sources.eye_source import EyeSource
from sources.wheel_speed_source import WheelSpeedSource
from sources.neural_trace_source import NeuralTraceSource
from sources.numpy_trace_source import NumpyTraceSource
from sources.sleep_score_source import SleepScoreSource


def _build_sources(template_sources, exp_id, user_id, exp_dir_processed):
    sources = {}
    for name, spec in template_sources.items():
        src_type = spec["type"]
        params = dict(spec["params"])
        if src_type in ("S2P_Binary", "VideoBinSource"):
            params["user"] = user_id
            params["expID"] = exp_id
            sources[name] = VideoBinSource(params)
        elif src_type == "StimulusVideoSource":
            cfg = {"user": user_id, "expID": exp_id}
            sources[name] = StimulusVideoSource(
                config=cfg,
                bonsai_root=params["bonsai_root"],
                stimulus_base_dir=params["stimulus_base_dir"],
                fps=int(params["fps"]),
            )
        elif src_type == "StimulusSource":
            cfg = {"user": user_id, "expID": exp_id}
            out_az_range = params.get("output_azimuth_range")
            out_el_range = params.get("output_elevation_range")
            sources[name] = StimulusSource(
                config=cfg,
                bonsai_root=str(params.get("bonsai_root", "D:\\bonsai_resources\\")),
                stimulus_base_dir=str(params.get("stimulus_base_dir", "/home/adamranson/data/vid_for_decoder/")),
                fps=int(params.get("fps", 30)),
                field_azimuth_range=tuple(params.get("field_azimuth_range", [-180.0, 180.0])),
                field_elevation_range=tuple(params.get("field_elevation_range", [-180.0, 180.0])),
                output_azimuth_center=float(params.get("output_azimuth_center", 0.0)),
                output_azimuth_span=float(params.get("output_azimuth_span", 360.0)),
                output_elevation_center=float(params.get("output_elevation_center", 0.0)),
                output_elevation_span=float(params.get("output_elevation_span", 360.0)),
                output_azimuth_range=tuple(out_az_range) if out_az_range else None,
                output_elevation_range=tuple(out_el_range) if out_el_range else None,
                pixels_per_degree=float(params.get("pixels_per_degree", 2.0)),
                background_gray=int(params.get("background_gray", 127)),
            )
        elif src_type == "ReconstructionVideoSource":
            video_path = params.get("video_path")
            timestamps_path = params.get("timestamps_path")
            edges_path = params.get("edges_path")
            if not video_path:
                subdir = params.get("subdir", "reconstruction")
                video_file = params.get("video_file", "session_recons_cut.mp4")
                video_path = os.path.join(subdir, video_file)
            if not timestamps_path:
                subdir = params.get("subdir", "reconstruction")
                timestamps_file = params.get("timestamps_file", "video_timeline.npy")
                timestamps_path = os.path.join(subdir, timestamps_file)
            video_path = (
                video_path
                if os.path.isabs(str(video_path))
                else os.path.join(exp_dir_processed, str(video_path))
            )
            timestamps_path = (
                timestamps_path
                if os.path.isabs(str(timestamps_path))
                else os.path.join(exp_dir_processed, str(timestamps_path))
            )
            resolved_edges_path = None
            if edges_path:
                resolved_edges_path = (
                    edges_path
                    if os.path.isabs(str(edges_path))
                    else os.path.join(exp_dir_processed, str(edges_path))
                )
            sources[name] = ReconstructionVideoSource(
                video_path=video_path,
                timestamps_path=timestamps_path,
                enable_temporal_filter=bool(params.get("enable_temporal_filter", False)),
                temporal_window=int(params.get("temporal_window", 0)),
                enable_spatial_filter=bool(params.get("enable_spatial_filter", False)),
                spatial_sigma=float(params.get("spatial_sigma", 0.0)),
                interpolate=bool(params.get("interpolate", False)),
                cache_size=int(params.get("cache_size", 128)),
                overlay_edges=bool(params.get("overlay_edges", False)),
                edges_path=resolved_edges_path,
            )
        elif src_type == "EyeSource":
            timestamps_file = params.get("timestamps_file", os.path.join("recordings", "eye_frame_times.npy"))
            timestamps_path = (
                timestamps_file
                if os.path.isabs(timestamps_file)
                else os.path.join(exp_dir_processed, timestamps_file)
            )
            crop_value = params.get("crop", "False")
            if isinstance(crop_value, str):
                if crop_value.lower() in ("false", "0", ""):
                    crop_value = False
                elif crop_value.lower() in ("true", "1"):
                    crop_value = True
            sources[name] = EyeSource(
                exp_dir_processed=exp_dir_processed,
                expID=exp_id,
                eye=str(params.get("eye", "right")),
                timestamps_path=timestamps_path,
                crop=crop_value,
                plot_detected_pupil=bool(params.get("plot_detected_pupil", False)),
                plot_detected_eye=bool(params.get("plot_detected_eye", False)),
                overlay_thickness=int(params.get("overlay_thickness", 2)),
                contrast_clip_percentiles=tuple(params.get("contrast_clip_percentiles", [])) or None,
            )
        elif src_type == "WheelSpeedSource":
            sources[name] = WheelSpeedSource(
                exp_dir_processed=exp_dir_processed,
                time_window=tuple(params.get("time_window", [-5.0, 5.0])),
                y_range_mode=str(params.get("y_range_mode", "global")),
                fixed_y_range=tuple(params.get("fixed_y_range", [])) or None,
                y_label=str(params.get("y_label", "")),
                title=str(params.get("title", "")),
                show_y_axis=bool(params.get("show_y_axis", True)),
                line_width=float(params.get("line_width", 1.5)),
                figure_size=tuple(params.get("figure_size", [4, 2])),
                dpi=int(params.get("dpi", 100)),
                bg_color=str(params.get("bg_color", "black")),
                grid=bool(params.get("grid", False)),
                font_color=str(params.get("font_color", "white")),
                interpolate=bool(params.get("interpolate", False)),
                colors=list(params.get("colors", ["cyan"])),
            )
        elif src_type == "NeuralTraceSource":
            sources[name] = NeuralTraceSource(
                exp_dir_processed=exp_dir_processed,
                channel=int(params.get("channel", 0)),
                signal_key=str(params.get("signal_key", "Spikes")),
                neuron_indices=list(params.get("neuron_indices", [])),
                time_window=tuple(params.get("time_window", [-5.0, 0.0])),
                y_range_mode=str(params.get("y_range_mode", "global")),
                fixed_y_range=tuple(params.get("fixed_y_range", [])) or None,
                y_label=str(params.get("y_label", "")),
                title=str(params.get("title", "")),
                show_y_axis=bool(params.get("show_y_axis", True)),
                line_width=float(params.get("line_width", 1.5)),
                figure_size=tuple(params.get("figure_size", [4, 2])),
                dpi=int(params.get("dpi", 100)),
                bg_color=str(params.get("bg_color", "black")),
                grid=bool(params.get("grid", False)),
                font_color=str(params.get("font_color", "white")),
                interpolate=bool(params.get("interpolate", False)),
                colors=list(params.get("colors", ["cyan"])),
            )
        elif src_type == "NumpyTraceSource":
            path = str(params.get("path", ""))
            if path and not os.path.isabs(path):
                path = os.path.join(exp_dir_processed, path)
            sources[name] = NumpyTraceSource(
                path=path,
                key=str(params.get("key", "")),
                columns=list(params.get("columns", [])),
                time_window=tuple(params.get("time_window", [-5.0, 0.0])),
                y_range_mode=str(params.get("y_range_mode", "global")),
                fixed_y_range=tuple(params.get("fixed_y_range", [])) or None,
                y_label=str(params.get("y_label", "")),
                title=str(params.get("title", "")),
                show_y_axis=bool(params.get("show_y_axis", True)),
                line_width=float(params.get("line_width", 1.5)),
                figure_size=tuple(params.get("figure_size", [4, 2])),
                dpi=int(params.get("dpi", 100)),
                bg_color=str(params.get("bg_color", "black")),
                grid=bool(params.get("grid", False)),
                font_color=str(params.get("font_color", "white")),
                interpolate=bool(params.get("interpolate", False)),
                colors=list(params.get("colors", ["cyan"])),
            )
        elif src_type == "SleepScoreSource":
            sources[name] = SleepScoreSource(
                exp_dir_processed=exp_dir_processed,
                time_window=tuple(params.get("time_window", [-5.0, 5.0])),
                colors=list(params.get("colors", ["#E76F51", "#F4A261", "#E9C46A", "#2A9D8F"])),
                labels=list(params.get("labels", ["AW", "QW", "NREM", "REM"])),
                line_width=float(params.get("line_width", 3.0)),
                figure_size=tuple(params.get("figure_size", [4, 2])),
                dpi=int(params.get("dpi", 100)),
                bg_color=str(params.get("bg_color", "black")),
                font_color=str(params.get("font_color", "white")),
                show_y_axis=bool(params.get("show_y_axis", True)),
            )
        else:
            raise ValueError(f"Unsupported source type: {src_type}")
    return sources


def main():
    parser = argparse.ArgumentParser(description="Export a video from a template.")
    parser.add_argument("--template", required=True)
    parser.add_argument("--expID", required=True)
    parser.add_argument("--userID", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--stop", type=float, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--play-fps", type=float, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--log", default="")
    args = parser.parse_args()

    if os.path.splitext(args.out)[1] == "":
        args.out = args.out + ".mp4"

    with open(args.template, "r", encoding="utf-8") as f:
        template = json.load(f)

    _animal_id, _remote, _processed_root, exp_dir_processed, _exp_dir_raw = organise_paths.find_paths(
        args.userID, args.expID
    )
    sources = _build_sources(template.get("sources", {}), args.expID, args.userID, exp_dir_processed)

    canvas = template.get("canvas", {})
    size = canvas.get("size", [800, 1200])
    layout_cfg = {
        "canvas_size": (int(size[0]), int(size[1])),
        "elements": {
            name: {
                "source": elem["source"],
                "x": elem["x"],
                "y": elem["y"],
                "w": elem["w"],
                "h": elem["h"],
            }
            for name, elem in template.get("elements", {}).items()
        },
    }

    composer = CanvasComposer(sources, layout_cfg, bg=int(canvas.get("bg", 0)))
    composer.initialize()
    sample_fps = args.fps
    play_fps = args.play_fps if args.play_fps is not None else sample_fps
    timeline = Timeline(args.start, args.stop, sample_fps)

    log_path = args.log.strip()
    def log(msg):
        if not log_path:
            return
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\\n")

    frame0 = composer.draw_composite(timeline.times[0])
    H, W = frame0.shape[:2]
    writer = VideoWriter(args.out, fps=play_fps, frame_size=(W, H))

    total = len(timeline)
    try:
        from tqdm import tqdm  # type: ignore
        for i, t in tqdm(list(enumerate(timeline)), total=total, unit="frame"):
            frame = composer.draw_composite(t)
            writer.write(frame)
            if total > 0:
                pct = (i + 1) / total * 100.0
                log(f"PROGRESS {pct:.2f}")
    except Exception:
        for i, t in enumerate(timeline):
            frame = composer.draw_composite(t)
            writer.write(frame)
            if total > 0:
                pct = (i + 1) / total * 100.0
                log(f"PROGRESS {pct:.2f}")
    writer.close()
    log("DONE")


if __name__ == "__main__":
    main()
