import argparse
import json
import os
import getpass
from typing import Any, Dict, List, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

import organise_paths
from core.canvas_composer import CanvasComposer
from core.timeline import Timeline
from core.writer import VideoWriter
from sources.video_bin_source import VideoBinSource
from sources.stimulus_video_source import StimulusVideoSource
from sources.reconstruction_video_source import ReconstructionVideoSource
from sources.eye_source import EyeSource
from sources.wheel_speed_source import WheelSpeedSource
from sources.neural_trace_source import NeuralTraceSource


PALETTE = [
    "#E76F51",
    "#F4A261",
    "#E9C46A",
    "#2A9D8F",
    "#264653",
    "#8E9AAF",
    "#CBC0D3",
    "#C9ADA7",
]


SOURCE_DEFS = {
    "S2P_Binary": {
        "params": [
            ("planes", "list[int]", [2, 4, 6, 8]),
            ("height", "int", 512),
            ("width", "int", 512),
            ("spatial_sigma", "float", 1.0),
            ("temporal_window", "int", 6),
            ("enable_spatial_filter", "bool", False),
            ("enable_temporal_filter", "bool", True),
            ("interpolate", "bool", True),
            ("tile_layout", "dict", {"rows": 2, "cols": 2, "order": [0, 1, 2, 3], "gap": 4}),
        ],
    },
    "StimulusVideoSource": {
        "params": [
            ("bonsai_root", "str", "D:\\bonsai_resources\\"),
            ("stimulus_base_dir", "str", "/home/adamranson/data/vid_for_decoder/"),
            ("fps", "int", 30),
        ],
    },
    "ReconstructionVideoSource": {
        "params": [
            ("subdir", "str", "reconstruction"),
            ("video_file", "str", "session_recons_cut.mp4"),
            ("timestamps_file", "str", "video_timeline.npy"),
            ("enable_temporal_filter", "bool", True),
            ("temporal_window", "int", 3),
            ("enable_spatial_filter", "bool", True),
            ("spatial_sigma", "float", 1.2),
            ("interpolate", "bool", True),
            ("cache_size", "int", 96),
            ("overlay_edges", "bool", False),
        ],
    },
    "EyeSource": {
        "params": [
            ("eye", "str", "right"),
            ("timestamps_file", "str", os.path.join("recordings", "eye_frame_times.npy")),
            ("crop", "crop", "False"),
            ("plot_detected_pupil", "bool", True),
            ("plot_detected_eye", "bool", True),
            ("overlay_thickness", "int", 2),
            ("contrast_clip_percentiles", "list[float]", [0.0, 90.0]),
        ],
    },
    "WheelSpeedSource": {
        "params": [
            ("time_window", "tuple[float]", [-5.0, 5.0]),
            ("y_range_mode", "str", "global"),
            ("fixed_y_range", "tuple[float]", []),
            ("y_label", "str", ""),
            ("title", "str", "Run speed"),
            ("show_y_axis", "bool", True),
            ("line_width", "float", 1.5),
            ("figure_size", "tuple[int]", [4, 2]),
            ("dpi", "int", 100),
            ("bg_color", "str", "black"),
            ("grid", "bool", False),
            ("font_color", "str", "white"),
            ("interpolate", "bool", True),
            ("colors", "list[str]", ["cyan"]),
        ],
    },
    "NeuralTraceSource": {
        "params": [
            ("channel", "int", 0),
            ("signal_key", "str", "Spikes"),
            ("neuron_indices", "list[int]", []),
            ("time_window", "tuple[float]", [-5.0, 0.0]),
            ("y_range_mode", "str", "global"),
            ("fixed_y_range", "tuple[float]", []),
            ("y_label", "str", ""),
            ("title", "str", "Mean population activity"),
            ("show_y_axis", "bool", True),
            ("line_width", "float", 1.5),
            ("figure_size", "tuple[int]", [4, 2]),
            ("dpi", "int", 100),
            ("bg_color", "str", "black"),
            ("grid", "bool", False),
            ("font_color", "str", "white"),
            ("interpolate", "bool", True),
            ("colors", "list[str]", ["cyan", "magenta"]),
        ],
    },
}


def _list_home_users() -> List[str]:
    try:
        entries = sorted(
            d for d in os.listdir("/home")
            if os.path.isdir(os.path.join("/home", d)) and not d.startswith(".")
        )
    except Exception:
        entries = []
    return entries


def _parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError("Invalid boolean")


def _parse_list(value: str) -> List[Any]:
    if value.strip() == "":
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return list(parsed)
    except Exception:
        pass
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts


def _parse_value(value: str, ftype: str):
    if ftype == "int":
        return int(value)
    if ftype == "float":
        return float(value)
    if ftype == "str":
        return value
    if ftype == "bool":
        return _parse_bool(value)
    if ftype == "list[int]":
        return [int(x) for x in _parse_list(value)]
    if ftype == "list[float]":
        return [float(x) for x in _parse_list(value)]
    if ftype == "list[str]":
        return [str(x) for x in _parse_list(value)]
    if ftype == "tuple[int]":
        vals = _parse_list(value)
        return [int(x) for x in vals]
    if ftype == "tuple[float]":
        vals = _parse_list(value)
        return [float(x) for x in vals]
    if ftype == "dict":
        return json.loads(value)
    if ftype == "crop":
        v = value.strip()
        if v == "":
            return False
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list) and len(parsed) == 4:
                return [int(x) for x in parsed]
        except Exception:
            pass
        return _parse_bool(v)
    return value


def _format_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return str(value)


class DraggableRect(QtWidgets.QGraphicsRectItem):
    def __init__(self, name: str, rect: QtCore.QRectF, on_moved, *args, **kwargs):
        super().__init__(rect, *args, **kwargs)
        self.name = name
        self.on_moved = on_moved
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            if self.on_moved:
                self.on_moved(self.name, value)
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event):
        if self.on_moved:
            self.on_moved(self.name, self.pos())
        super().mouseReleaseEvent(event)


class CanvasView(QtWidgets.QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QtGui.QColor("#1B1F24"))
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)


class ComposerWindow(QtWidgets.QMainWindow):
    def __init__(self, template_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("Experiment Composer")
        self.resize(1500, 950)

        self.canvas_w = 500
        self.canvas_h = 500
        self.bg = 0

        self.sources: Dict[str, Dict[str, Any]] = {}
        self.elements: Dict[str, Dict[str, Any]] = {}
        self.selected_source: Optional[str] = None

        self.template_path = None
        self._composer = None
        self._timeline = None

        self._color_index = 0
        self._item_map: Dict[str, QtWidgets.QGraphicsRectItem] = {}
        self._rendering = False

        self._build_ui()
        if template_path:
            self._load_template_file(template_path)
        self._render_canvas()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(left_panel)
        scroll_area.setFixedWidth(360)
        layout.addWidget(scroll_area)

        self.scene = QtWidgets.QGraphicsScene(0, 0, self.canvas_w, self.canvas_h)
        self.scene.selectionChanged.connect(self._on_scene_selection)
        self.view = CanvasView(self.scene)
        layout.addWidget(self.view, stretch=1)

        # Experiment
        exp_group = QtWidgets.QGroupBox("Experiment")
        exp_form = QtWidgets.QFormLayout(exp_group)
        self.user_combo = QtWidgets.QComboBox()
        users = _list_home_users()
        current_user = getpass.getuser()
        if current_user not in users:
            users.insert(0, current_user)
        self.user_combo.addItems(users)
        self.user_combo.setCurrentText("pmateosaparicio")
        self.exp_id_edit = QtWidgets.QLineEdit("2025-07-04_06_ESPM154")
        exp_form.addRow("User", self.user_combo)
        exp_form.addRow("Exp ID", self.exp_id_edit)
        left_layout.addWidget(exp_group)

        # Canvas
        canvas_group = QtWidgets.QGroupBox("Canvas")
        canvas_form = QtWidgets.QFormLayout(canvas_group)
        self.canvas_w_edit = QtWidgets.QLineEdit(str(self.canvas_w))
        self.canvas_h_edit = QtWidgets.QLineEdit(str(self.canvas_h))
        canvas_form.addRow("Width", self.canvas_w_edit)
        canvas_form.addRow("Height", self.canvas_h_edit)
        self.canvas_apply_btn = QtWidgets.QPushButton("Apply Canvas Size")
        self.canvas_apply_btn.clicked.connect(self._apply_canvas_size)
        canvas_form.addRow(self.canvas_apply_btn)
        left_layout.addWidget(canvas_group)

        # Source add
        add_group = QtWidgets.QGroupBox("Add Source")
        add_form = QtWidgets.QFormLayout(add_group)
        self.source_type_combo = QtWidgets.QComboBox()
        self.source_type_combo.addItems(sorted(SOURCE_DEFS.keys()))
        self.source_type_combo.currentTextChanged.connect(self._update_default_name)
        self.source_name_edit = QtWidgets.QLineEdit()
        self.source_name_edit.textEdited.connect(self._on_name_edited)
        self._name_dirty = False
        self.add_source_btn = QtWidgets.QPushButton("Add")
        self.add_source_btn.clicked.connect(self._add_source)
        add_form.addRow("Type", self.source_type_combo)
        add_form.addRow("Name", self.source_name_edit)
        add_form.addRow(self.add_source_btn)
        left_layout.addWidget(add_group)
        self._update_default_name(self.source_type_combo.currentText())

        # Sources list
        self.sources_list = QtWidgets.QListWidget()
        self.sources_list.currentTextChanged.connect(self._on_source_select)
        left_layout.addWidget(QtWidgets.QLabel("Sources"))
        left_layout.addWidget(self.sources_list)

        self.delete_source_btn = QtWidgets.QPushButton("Delete Source")
        self.delete_source_btn.clicked.connect(self._delete_source)
        left_layout.addWidget(self.delete_source_btn)

        # Params
        self.params_group = QtWidgets.QGroupBox("Source Params")
        self.params_form = QtWidgets.QFormLayout(self.params_group)
        self.params_widgets: Dict[str, QtWidgets.QLineEdit] = {}
        self.params_apply_btn = QtWidgets.QPushButton("Apply Params")
        self.params_apply_btn.clicked.connect(self._apply_source_params)
        self.params_form.addRow(self.params_apply_btn)
        left_layout.addWidget(self.params_group)

        # Layout
        layout_group = QtWidgets.QGroupBox("Layout")
        layout_form = QtWidgets.QFormLayout(layout_group)
        self.x_edit = QtWidgets.QLineEdit()
        self.y_edit = QtWidgets.QLineEdit()
        self.w_edit = QtWidgets.QLineEdit()
        self.h_edit = QtWidgets.QLineEdit()
        layout_form.addRow("X", self.x_edit)
        layout_form.addRow("Y", self.y_edit)
        layout_form.addRow("W", self.w_edit)
        layout_form.addRow("H", self.h_edit)
        self.layout_apply_btn = QtWidgets.QPushButton("Apply Layout")
        self.layout_apply_btn.clicked.connect(self._apply_layout)
        layout_form.addRow(self.layout_apply_btn)
        left_layout.addWidget(layout_group)

        # Template + Export
        template_group = QtWidgets.QGroupBox("Template")
        template_layout = QtWidgets.QVBoxLayout(template_group)
        template_btn_row = QtWidgets.QHBoxLayout()
        self.save_template_btn = QtWidgets.QPushButton("Save")
        self.load_template_btn = QtWidgets.QPushButton("Load")
        self.save_template_btn.clicked.connect(self._save_template)
        self.load_template_btn.clicked.connect(self._load_template)
        template_btn_row.addWidget(self.save_template_btn)
        template_btn_row.addWidget(self.load_template_btn)
        template_layout.addLayout(template_btn_row)

        export_form = QtWidgets.QFormLayout()
        self.export_start_edit = QtWidgets.QLineEdit("0")
        self.export_stop_edit = QtWidgets.QLineEdit("10")
        self.export_fps_edit = QtWidgets.QLineEdit("20")
        self.export_tmux_check = QtWidgets.QCheckBox("Export in tmux")
        self.export_btn = QtWidgets.QPushButton("Export Video")
        self.export_btn.clicked.connect(self._export_video)
        export_form.addRow("Start", self.export_start_edit)
        export_form.addRow("Stop", self.export_stop_edit)
        export_form.addRow("FPS", self.export_fps_edit)
        export_form.addRow(self.export_tmux_check)
        export_form.addRow(self.export_btn)
        template_layout.addLayout(export_form)
        left_layout.insertWidget(1, template_group)

        left_layout.addStretch(1)

    def _next_color(self):
        color = PALETTE[self._color_index % len(PALETTE)]
        self._color_index += 1
        return color

    def _apply_canvas_size(self):
        try:
            self.canvas_w = int(self.canvas_w_edit.text())
            self.canvas_h = int(self.canvas_h_edit.text())
        except ValueError:
            self._error("Canvas size must be integers.")
            return
        self.scene.setSceneRect(0, 0, self.canvas_w, self.canvas_h)
        self._render_canvas()

    def _render_canvas(self):
        self._rendering = True
        self.scene.clear()
        self._item_map.clear()
        border_pen = QtGui.QPen(QtGui.QColor("#8A8A8A"), 1, QtCore.Qt.PenStyle.DashLine)
        border_pen.setCosmetic(True)
        border_rect = QtWidgets.QGraphicsRectItem(0, 0, self.canvas_w, self.canvas_h)
        border_rect.setPen(border_pen)
        border_rect.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush))
        border_rect.setZValue(-10)
        self.scene.addItem(border_rect)
        for name, elem in self.elements.items():
            x = int(elem.get("x", 0))
            y = int(elem.get("y", 0))
            w = int(elem.get("w", 0))
            h = int(elem.get("h", 0))
            rect = QtCore.QRectF(0, 0, w, h)
            item = DraggableRect(name, rect, self._on_item_moved)
            item.setBrush(QtGui.QBrush(QtGui.QColor(elem.get("color", "#666666"))))
            item.setPen(QtGui.QPen(QtGui.QColor("white"), 1))
            item.setPos(x, y)
            self.scene.addItem(item)
            self._item_map[name] = item

            label = QtWidgets.QGraphicsTextItem(f"{name}\n[{elem['source']}]", item)
            label.setDefaultTextColor(QtGui.QColor("white"))
            label.setPos(6, 6)
        self.scene.update()
        if self.selected_source and self.selected_source in self._item_map:
            self._item_map[self.selected_source].setSelected(True)
        self._rendering = False

    def _add_source(self):
        src_type = self.source_type_combo.currentText()
        name = self.source_name_edit.text().strip()
        if not name:
            base = src_type
            idx = 1
            name = f"{base}_{idx}"
            while name in self.sources or name in self.elements:
                idx += 1
                name = f"{base}_{idx}"
        if name in self.sources or name in self.elements:
            self._error(f"Name '{name}' already exists.")
            return
        params = {k: v for k, _, v in SOURCE_DEFS[src_type]["params"]}
        self.sources[name] = {"type": src_type, "params": params}
        self.elements[name] = {
            "source": name,
            "x": 0,
            "y": 0,
            "w": 200,
            "h": 150,
            "color": self._next_color(),
        }
        self.source_name_edit.clear()
        self._name_dirty = False
        self._update_default_name(self.source_type_combo.currentText())
        self._refresh_sources_list()
        self.sources_list.setCurrentRow(self.sources_list.count() - 1)
        self._render_canvas()

    def _delete_source(self):
        name = self.selected_source
        if not name:
            return
        if name in self.elements:
            del self.elements[name]
        if name in self.sources:
            del self.sources[name]
        self.selected_source = None
        self._refresh_sources_list()
        self._clear_params()
        self._clear_layout_fields()
        self._render_canvas()

    def _refresh_sources_list(self):
        self.sources_list.blockSignals(True)
        self.sources_list.clear()
        for name in sorted(self.sources.keys()):
            self.sources_list.addItem(name)
        self.sources_list.blockSignals(False)

    def _on_source_select(self, name: str):
        if not name:
            return
        self.selected_source = name
        self._load_source_params()
        self._load_layout_fields()
        item = self._item_map.get(name)
        if item:
            self.scene.clearSelection()
            item.setSelected(True)

    def _on_scene_selection(self):
        items = self.scene.selectedItems()
        if not items:
            return
        item = items[0]
        name = getattr(item, "name", None)
        if not name:
            return
        self.selected_source = name
        self._load_source_params()
        self._load_layout_fields()
        self.sources_list.setCurrentRow(self._index_of_source(name))

    def _on_item_moved(self, name: str, pos: QtCore.QPointF):
        if self._rendering:
            return
        elem = self.elements.get(name)
        if not elem:
            return
        x = int(max(0, min(self.canvas_w - elem["w"], pos.x())))
        y = int(max(0, min(self.canvas_h - elem["h"], pos.y())))
        elem["x"] = x
        elem["y"] = y
        item = self._item_map.get(name)
        if item and (item.pos().x() != x or item.pos().y() != y):
            item.blockSignals(True)
            item.setPos(x, y)
            item.blockSignals(False)
        if self.selected_source != name:
            self.selected_source = name
            self.sources_list.setCurrentRow(self._index_of_source(name))
        self.x_edit.setText(str(elem["x"]))
        self.y_edit.setText(str(elem["y"]))
        self.w_edit.setText(str(elem["w"]))
        self.h_edit.setText(str(elem["h"]))

    def _index_of_source(self, name: str) -> int:
        for i in range(self.sources_list.count()):
            if self.sources_list.item(i).text() == name:
                return i
        return -1

    def _clear_params(self):
        if self.params_apply_btn:
            self.params_form.removeWidget(self.params_apply_btn)
        while self.params_form.rowCount() > 0:
            self.params_form.removeRow(0)
        if self.params_apply_btn:
            self.params_form.addRow(self.params_apply_btn)
        self.params_widgets = {}

    def _load_source_params(self):
        self._clear_params()
        if not self.selected_source:
            return
        src = self.sources[self.selected_source]
        src_type = src["type"]
        params = src["params"]
        for key, ftype, default in SOURCE_DEFS[src_type]["params"]:
            edit = QtWidgets.QLineEdit(_format_value(params.get(key, default)))
            self.params_form.insertRow(self.params_form.rowCount() - 1, key, edit)
            self.params_widgets[key] = edit

    def _apply_source_params(self):
        if not self.selected_source:
            return
        src = self.sources[self.selected_source]
        src_type = src["type"]
        for key, ftype, _default in SOURCE_DEFS[src_type]["params"]:
            edit = self.params_widgets.get(key)
            if not edit:
                continue
            raw = edit.text().strip()
            if raw == "" and ftype in ("list[int]", "list[float]", "list[str]", "tuple[int]", "tuple[float]"):
                value = []
            elif raw == "" and ftype == "dict":
                value = {}
            else:
                try:
                    value = _parse_value(raw, ftype)
                except Exception as e:
                    self._error(f"{key}: {e}")
                    return
            src["params"][key] = value

    def _load_layout_fields(self):
        if not self.selected_source:
            return
        elem = self.elements.get(self.selected_source)
        if not elem:
            return
        self.x_edit.setText(str(elem["x"]))
        self.y_edit.setText(str(elem["y"]))
        self.w_edit.setText(str(elem["w"]))
        self.h_edit.setText(str(elem["h"]))

    def _clear_layout_fields(self):
        self.x_edit.clear()
        self.y_edit.clear()
        self.w_edit.clear()
        self.h_edit.clear()

    def _apply_layout(self):
        if not self.selected_source:
            return
        elem = self.elements.get(self.selected_source)
        if not elem:
            return
        try:
            x = int(self.x_edit.text())
            y = int(self.y_edit.text())
            w = int(self.w_edit.text())
            h = int(self.h_edit.text())
        except ValueError:
            self._error("Layout values must be integers.")
            return
        if w <= 0 or h <= 0:
            self._error("Width and height must be > 0.")
            return
        if x < 0 or y < 0 or x + w > self.canvas_w or y + h > self.canvas_h:
            self._error("Element must fit within the canvas bounds.")
            return
        elem["x"] = x
        elem["y"] = y
        elem["w"] = w
        elem["h"] = h
        self._render_canvas()

    def _on_name_edited(self):
        self._name_dirty = True

    def _update_default_name(self, src_type: str):
        if self._name_dirty and self.source_name_edit.text().strip():
            return
        base = src_type
        idx = 1
        name = f"{base}_{idx}"
        while name in self.sources or name in self.elements:
            idx += 1
            name = f"{base}_{idx}"
        self.source_name_edit.setText(name)

    def _save_template(self):
        template = {
            "timeline": {
                "start": float(self.export_start_edit.text() or 0),
                "stop": float(self.export_stop_edit.text() or 0),
                "fps": float(self.export_fps_edit.text() or 0),
            },
            "canvas": {"size": [self.canvas_h, self.canvas_w], "bg": self.bg},
            "sources": self.sources,
            "elements": {
                name: {
                    "source": elem["source"],
                    "x": elem["x"],
                    "y": elem["y"],
                    "w": elem["w"],
                    "h": elem["h"],
                    "color": elem.get("color", ""),
                }
                for name, elem in self.elements.items()
            },
        }
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Template", "", "JSON (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2)
        self.template_path = path

    def _load_template(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Template", "", "JSON (*.json)")
        if not path:
            return
        self._load_template_file(path)

    def _load_template_file(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            template = json.load(f)
        timeline = template.get("timeline", {})
        self.export_start_edit.setText(str(timeline.get("start", 0)))
        self.export_stop_edit.setText(str(timeline.get("stop", 10)))
        self.export_fps_edit.setText(str(timeline.get("fps", 20)))

        canvas = template.get("canvas", {})
        size = canvas.get("size", [self.canvas_h, self.canvas_w])
        self.canvas_h = int(size[0])
        self.canvas_w = int(size[1])
        self.bg = int(canvas.get("bg", 0))
        self.canvas_w_edit.setText(str(self.canvas_w))
        self.canvas_h_edit.setText(str(self.canvas_h))
        self.scene.setSceneRect(0, 0, self.canvas_w, self.canvas_h)

        self.sources = template.get("sources", {})
        self.elements = template.get("elements", {})
        self.selected_source = None
        self._refresh_sources_list()
        self._render_canvas()

    def _build_composer(self, exp_id: str, user_id: str, t_start: float, t_stop: float, t_fps: float, progress_cb=None):
        timeline = Timeline(t_start, t_stop, t_fps)
        _animal_id, _remote, _processed_root, exp_dir_processed, _exp_dir_raw = organise_paths.find_paths(user_id, exp_id)

        sources = {}
        for name, spec in self.sources.items():
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
            elif src_type == "ReconstructionVideoSource":
                subdir = params.get("subdir", "reconstruction")
                video_file = params.get("video_file", "session_recons_cut.mp4")
                timestamps_file = params.get("timestamps_file", "video_timeline.npy")
                video_path = os.path.join(exp_dir_processed, subdir, video_file)
                timestamps_path = os.path.join(exp_dir_processed, subdir, timestamps_file)
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
            else:
                raise ValueError(f"Unsupported source type: {src_type}")

        layout_cfg = {
            "canvas_size": (self.canvas_h, self.canvas_w),
            "elements": {
                name: {
                    "source": elem["source"],
                    "x": elem["x"],
                    "y": elem["y"],
                    "w": elem["w"],
                    "h": elem["h"],
                }
                for name, elem in self.elements.items()
            },
        }
        composer = CanvasComposer(sources, layout_cfg, bg=self.bg)
        if progress_cb:
            total = max(1, len(sources))
            for idx, (name, src) in enumerate(sources.items(), start=1):
                progress_cb(f"Initializing {name} ({idx}/{total})", idx - 1, total)
                src.initialize()
            progress_cb("Finalizing", total, total)
        else:
            composer.initialize()
        return composer, timeline

    def _load_experiment(self):
        exp_id = self.exp_id_edit.text().strip()
        user_id = self.user_combo.currentText().strip()
        if not exp_id:
            self._error("Enter an experiment ID.")
            return
        if not self.sources or not self.elements:
            self._error("Add at least one source.")
            return
        try:
            t_start = float(self.export_start_edit.text())
            t_stop = float(self.export_stop_edit.text())
            t_fps = float(self.export_fps_edit.text())
        except ValueError:
            self._error("Timeline values must be numeric.")
            return

        progress = QtWidgets.QProgressDialog("Loading sources...", None, 0, 0, self)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.show()

        def update_progress(message, value, maximum):
            progress.setLabelText(message)
            progress.setMaximum(maximum)
            progress.setValue(value)
            QtWidgets.QApplication.processEvents()

        try:
            self._composer, self._timeline = self._build_composer(
                exp_id,
                user_id,
                t_start,
                t_stop,
                t_fps,
                progress_cb=update_progress,
            )
        except Exception as e:
            progress.close()
            self._error(str(e))
            return
        progress.close()

    def _export_video(self):
        exp_id = self.exp_id_edit.text().strip()
        user_id = self.user_combo.currentText().strip()
        if not exp_id:
            self._error("Enter an experiment ID.")
            return
        if not self.sources or not self.elements:
            self._error("Add at least one source.")
            return
        try:
            t_start = float(self.export_start_edit.text())
            t_stop = float(self.export_stop_edit.text())
            fps = float(self.export_fps_edit.text())
        except ValueError:
            self._error("Export values must be numeric.")
            return

        default_dir = os.path.join(os.getcwd(), "output_videos")
        os.makedirs(default_dir, exist_ok=True)
        timestamp = QtCore.QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
        default_name = f"{exp_id}_{timestamp}.mp4"
        default_path = os.path.join(default_dir, default_name)
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Video", default_path, "MP4 (*.mp4)")
        if not out_path:
            return
        out_path = self._ensure_mp4_path(out_path)

        if self.export_tmux_check.isChecked():
            if not self.template_path:
                self._error("Save a template before exporting in tmux.")
                return
            cmd = (
                f"python export_from_template.py "
                f"--template \"{self.template_path}\" "
                f"--expID \"{exp_id}\" --userID \"{user_id}\" "
                f"--start {t_start} --stop {t_stop} --fps {fps} "
                f"--out \"{out_path}\""
            )
            session_name = f"composer_export_{os.getpid()}"
            os.system(f"tmux new-session -d -s {session_name} \"{cmd}\"")
            QtWidgets.QMessageBox.information(self, "Export started", f"tmux session: {session_name}")
            return

        try:
            composer, timeline = self._build_composer(exp_id, user_id, t_start, t_stop, fps)
        except Exception as e:
            self._error(str(e))
            return

        try:
            frame0 = composer.draw_composite(timeline.times[0])
            H, W = frame0.shape[:2]
            writer = VideoWriter(out_path, fps=timeline.fps, frame_size=(W, H))
            progress = QtWidgets.QProgressDialog("Exporting frames...", None, 0, len(timeline), self)
            progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
            progress.show()
            for i, t in enumerate(timeline):
                frame = composer.draw_composite(t)
                writer.write(frame)
                progress.setValue(i + 1)
                QtWidgets.QApplication.processEvents()
            writer.close()
            progress.close()
        except Exception as e:
            self._error(str(e))

    def _error(self, msg: str):
        QtWidgets.QMessageBox.critical(self, "Error", msg)

    def _ensure_mp4_path(self, path: str) -> str:
        root, ext = os.path.splitext(path)
        if ext == "":
            return root + ".mp4"
        return path


def main():
    parser = argparse.ArgumentParser(description="GUI for composing experiment layouts.")
    parser.add_argument("--load", help="Path to a template JSON to load.", default=None)
    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    window = ComposerWindow(template_path=args.load)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
