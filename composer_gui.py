import argparse
import json
import tkinter as tk
from tkinter import filedialog, messagebox


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


class LayoutComposerGUI:
    def __init__(self, root, layout_cfg=None):
        self.root = root
        self.root.title("Experiment Composer - Layout GUI")

        self.canvas_w = 1200
        self.canvas_h = 800
        self.elements = {}
        self.selected = None
        self.drag_offset = (0, 0)
        self.color_index = 0

        if layout_cfg:
            self._load_layout_cfg(layout_cfg)

        self._build_ui()
        self._render_all()

    def _load_layout_cfg(self, layout_cfg):
        canvas_size = layout_cfg.get("canvas_size", (self.canvas_h, self.canvas_w))
        if isinstance(canvas_size, (list, tuple)) and len(canvas_size) == 2:
            self.canvas_h, self.canvas_w = int(canvas_size[0]), int(canvas_size[1])
        for name, elem in layout_cfg.get("elements", {}).items():
            self.elements[name] = {
                "source": elem.get("source", ""),
                "x": int(elem.get("x", 0)),
                "y": int(elem.get("y", 0)),
                "w": int(elem.get("w", 100)),
                "h": int(elem.get("h", 100)),
                "color": self._next_color(),
            }

    def _build_ui(self):
        self.root.geometry("1400x900")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)

        panel = tk.Frame(self.root, padx=10, pady=10)
        panel.grid(row=0, column=0, sticky="ns")

        tk.Label(panel, text="Canvas").grid(row=0, column=0, columnspan=2, sticky="w")
        tk.Label(panel, text="Width").grid(row=1, column=0, sticky="w")
        tk.Label(panel, text="Height").grid(row=2, column=0, sticky="w")

        self.entry_canvas_w = tk.Entry(panel, width=10)
        self.entry_canvas_h = tk.Entry(panel, width=10)
        self.entry_canvas_w.grid(row=1, column=1, sticky="w")
        self.entry_canvas_h.grid(row=2, column=1, sticky="w")
        self.entry_canvas_w.insert(0, str(self.canvas_w))
        self.entry_canvas_h.insert(0, str(self.canvas_h))

        tk.Button(panel, text="Apply Canvas Size", command=self._apply_canvas_size).grid(
            row=3, column=0, columnspan=2, pady=(4, 10), sticky="ew"
        )

        tk.Label(panel, text="Elements").grid(row=4, column=0, columnspan=2, sticky="w")
        self.listbox = tk.Listbox(panel, width=26, height=12)
        self.listbox.grid(row=5, column=0, columnspan=2, sticky="w")
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)

        tk.Button(panel, text="Add", command=self._add_element).grid(row=6, column=0, sticky="ew")
        tk.Button(panel, text="Delete", command=self._delete_element).grid(row=6, column=1, sticky="ew")

        tk.Label(panel, text="Name").grid(row=7, column=0, sticky="w")
        tk.Label(panel, text="Source").grid(row=8, column=0, sticky="w")
        tk.Label(panel, text="X").grid(row=9, column=0, sticky="w")
        tk.Label(panel, text="Y").grid(row=10, column=0, sticky="w")
        tk.Label(panel, text="W").grid(row=11, column=0, sticky="w")
        tk.Label(panel, text="H").grid(row=12, column=0, sticky="w")

        self.entry_name = tk.Entry(panel, width=18)
        self.entry_source = tk.Entry(panel, width=18)
        self.entry_x = tk.Entry(panel, width=8)
        self.entry_y = tk.Entry(panel, width=8)
        self.entry_w = tk.Entry(panel, width=8)
        self.entry_h = tk.Entry(panel, width=8)

        self.entry_name.grid(row=7, column=1, sticky="w")
        self.entry_source.grid(row=8, column=1, sticky="w")
        self.entry_x.grid(row=9, column=1, sticky="w")
        self.entry_y.grid(row=10, column=1, sticky="w")
        self.entry_w.grid(row=11, column=1, sticky="w")
        self.entry_h.grid(row=12, column=1, sticky="w")

        tk.Button(panel, text="Apply Element", command=self._apply_element).grid(
            row=13, column=0, columnspan=2, pady=(6, 10), sticky="ew"
        )

        tk.Button(panel, text="Save Layout JSON", command=self._save_layout).grid(
            row=14, column=0, columnspan=2, sticky="ew"
        )
        tk.Button(panel, text="Print Python Config", command=self._print_python).grid(
            row=15, column=0, columnspan=2, sticky="ew"
        )

        self.canvas = tk.Canvas(self.root, bg="#1B1F24")
        self.canvas.grid(row=0, column=1, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_down)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_up)

    def _next_color(self):
        color = PALETTE[self.color_index % len(PALETTE)]
        self.color_index += 1
        return color

    def _refresh_listbox(self):
        self.listbox.delete(0, tk.END)
        for name in sorted(self.elements.keys()):
            self.listbox.insert(tk.END, name)

    def _render_all(self):
        self.canvas.config(width=self.canvas_w, height=self.canvas_h)
        self.canvas.delete("all")
        for name in self.elements:
            self._draw_element(name)
        self._refresh_listbox()

    def _draw_element(self, name):
        elem = self.elements[name]
        x, y, w, h = elem["x"], elem["y"], elem["w"], elem["h"]
        color = elem["color"]
        rect = self.canvas.create_rectangle(
            x,
            y,
            x + w,
            y + h,
            outline="white" if name == self.selected else "#AAAAAA",
            width=2 if name == self.selected else 1,
            fill=color,
            stipple="gray50",
            tags=(f"elem:{name}", "element"),
        )
        self.canvas.create_text(
            x + 6,
            y + 6,
            anchor="nw",
            text=f"{name}\n[{elem['source']}]",
            fill="white",
            font=("Helvetica", 10, "bold"),
            tags=(f"elem:{name}", "element"),
        )
        return rect

    def _apply_canvas_size(self):
        try:
            self.canvas_w = int(self.entry_canvas_w.get())
            self.canvas_h = int(self.entry_canvas_h.get())
        except ValueError:
            messagebox.showerror("Invalid size", "Canvas width/height must be integers.")
            return
        self._render_all()

    def _add_element(self):
        base_name = "element"
        idx = 1
        while f"{base_name}{idx}" in self.elements:
            idx += 1
        name = f"{base_name}{idx}"
        self.elements[name] = {
            "source": "",
            "x": 0,
            "y": 0,
            "w": 200,
            "h": 150,
            "color": self._next_color(),
        }
        self.selected = name
        self._render_all()
        self._load_selected_into_fields()

    def _delete_element(self):
        if not self.selected:
            return
        del self.elements[self.selected]
        self.selected = None
        self._render_all()

    def _on_list_select(self, _event):
        selection = self.listbox.curselection()
        if not selection:
            return
        name = self.listbox.get(selection[0])
        self.selected = name
        self._render_all()
        self._load_selected_into_fields()

    def _load_selected_into_fields(self):
        if not self.selected:
            return
        elem = self.elements[self.selected]
        self.entry_name.delete(0, tk.END)
        self.entry_source.delete(0, tk.END)
        self.entry_x.delete(0, tk.END)
        self.entry_y.delete(0, tk.END)
        self.entry_w.delete(0, tk.END)
        self.entry_h.delete(0, tk.END)
        self.entry_name.insert(0, self.selected)
        self.entry_source.insert(0, elem["source"])
        self.entry_x.insert(0, str(elem["x"]))
        self.entry_y.insert(0, str(elem["y"]))
        self.entry_w.insert(0, str(elem["w"]))
        self.entry_h.insert(0, str(elem["h"]))

    def _apply_element(self):
        if not self.selected:
            return
        name = self.entry_name.get().strip()
        if not name:
            messagebox.showerror("Invalid name", "Element name cannot be empty.")
            return
        if name != self.selected and name in self.elements:
            messagebox.showerror("Duplicate name", f"Element '{name}' already exists.")
            return
        try:
            x = int(self.entry_x.get())
            y = int(self.entry_y.get())
            w = int(self.entry_w.get())
            h = int(self.entry_h.get())
        except ValueError:
            messagebox.showerror("Invalid values", "X, Y, W, H must be integers.")
            return
        if w <= 0 or h <= 0:
            messagebox.showerror("Invalid size", "Width and height must be > 0.")
            return
        if x < 0 or y < 0 or x + w > self.canvas_w or y + h > self.canvas_h:
            messagebox.showerror(
                "Out of bounds",
                "Element must fit within the canvas bounds.",
            )
            return

        elem = self.elements.pop(self.selected)
        elem["source"] = self.entry_source.get().strip()
        elem["x"] = x
        elem["y"] = y
        elem["w"] = w
        elem["h"] = h
        self.elements[name] = elem
        self.selected = name
        self._render_all()
        self._load_selected_into_fields()

    def _element_from_canvas_item(self, item_id):
        tags = self.canvas.gettags(item_id)
        for tag in tags:
            if tag.startswith("elem:"):
                return tag.split(":", 1)[1]
        return None

    def _on_canvas_down(self, event):
        items = self.canvas.find_overlapping(event.x, event.y, event.x, event.y)
        name = None
        for item in items:
            name = self._element_from_canvas_item(item)
            if name:
                break
        if not name:
            return
        self.selected = name
        elem = self.elements[name]
        self.drag_offset = (event.x - elem["x"], event.y - elem["y"])
        self._render_all()
        self._load_selected_into_fields()

    def _on_canvas_drag(self, event):
        if not self.selected:
            return
        elem = self.elements[self.selected]
        dx, dy = self.drag_offset
        new_x = max(0, min(self.canvas_w - elem["w"], event.x - dx))
        new_y = max(0, min(self.canvas_h - elem["h"], event.y - dy))
        elem["x"] = new_x
        elem["y"] = new_y
        self._render_all()
        self._load_selected_into_fields()

    def _on_canvas_up(self, _event):
        pass

    def _save_layout(self):
        layout = self._current_layout_cfg()
        path = filedialog.asksaveasfilename(
            title="Save layout JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(layout, f, indent=2)
        messagebox.showinfo("Saved", f"Layout saved to {path}")

    def _print_python(self):
        layout = self._current_layout_cfg()
        print("layout_cfg = " + json.dumps(layout, indent=4))

    def _current_layout_cfg(self):
        return {
            "canvas_size": [self.canvas_h, self.canvas_w],
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


def _load_layout_from_path(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="GUI for composing experiment layouts.")
    parser.add_argument("--load", help="Path to a layout JSON to load.", default=None)
    args = parser.parse_args()

    layout_cfg = None
    if args.load:
        layout_cfg = _load_layout_from_path(args.load)

    root = tk.Tk()
    LayoutComposerGUI(root, layout_cfg=layout_cfg)
    root.mainloop()


if __name__ == "__main__":
    main()
