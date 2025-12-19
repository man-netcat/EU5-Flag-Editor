#!/usr/bin/env python3
"""EU5 Flag Editor - simple PySide6 app
See README.md for usage/requirements.
"""
import sys
import os
import json
from pathlib import Path
from functools import partial

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QPushButton,
    QFileDialog,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QSplitter,
    QSlider,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QInputDialog,
    QColorDialog,
    QGraphicsItem,
    QAbstractItemView,
    QSpinBox,
)
from PySide6.QtGui import QPixmap, QImage, Qt, QDrag, QPainter, QColor
from PySide6.QtCore import QByteArray, QMimeData, QPointF, QSize

from PIL import Image, ImageOps
import re
import colorsys
import math


class DraggableListWidget(QListWidget):
    def __init__(self, asset_type: str = "colored_emblems", *args, **kwargs):
        super().__init__(*args, **kwargs)
        # asset_type is the key in MainWindow.assets this list represents
        self.asset_type = asset_type

    def startDrag(self, supportedActions):
        item = self.currentItem()
        if not item:
            return
        name = item.text()
        # parent MainWindow stores assets dict; walk up to find it
        mw = self.window()
        path = None
        if hasattr(mw, "assets") and self.asset_type in mw.assets:
            path = mw.assets[self.asset_type].get(name)
        if not path:
            return
        # debug: starting drag suppressed
        drag = QDrag(self)
        md = QMimeData()
        payload = json.dumps({"path": str(path), "type": "colored"})
        # set type according to asset_type
        p = json.loads(payload)
        if self.asset_type == "textured_emblems":
            p["type"] = "textured"
        else:
            p["type"] = "colored"
        md.setData("application/x-eu5-asset", QByteArray(json.dumps(p).encode("utf-8")))
        drag.setMimeData(md)
        img = load_image(path)
        drag.setPixmap(pil2pixmap(img.resize((64, 64))))
        drag.exec(Qt.CopyAction)

    def mousePressEvent(self, event):
        # Let the base class process the event first (selection/drag logic)
        super().mousePressEvent(event)
        try:
            item = self.itemAt(event.pos())
            # only update current item if it's different to avoid unnecessary scrolling
            if item and item is not self.currentItem():
                self.setCurrentItem(item)
        except Exception:
            pass


class DropGraphicsView(QGraphicsView):
    def __init__(self, scene, owner=None):
        super().__init__(scene)
        self.owner = owner
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-eu5-asset"):
            # debug suppressed
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat("application/x-eu5-asset"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not event.mimeData().hasFormat("application/x-eu5-asset"):
            event.ignore()
            return
        try:
            md = event.mimeData().data("application/x-eu5-asset")
            payload = json.loads(bytes(md).decode("utf-8"))
            path = Path(payload["path"])
            # drop payload received
            typ = payload.get("type", "colored")
            # compute drop scene position
            if hasattr(event, "position"):
                posf = event.position()
                vx, vy = posf.x(), posf.y()
            else:
                vp = event.pos()
                vx, vy = vp.x(), vp.y()
            scene_pos = self.mapToScene(int(vx), int(vy))
            if self.owner:
                self.owner.add_emblem_to_scene(path, typ, scene_pos)
                event.acceptProposedAction()
            else:
                # fallback: directly add
                pil = load_image(path)
                it = EmblemItem(path, pil, texture_type=typ)
                it.setPos(
                    scene_pos
                    - QPointF(it.pixmap().width() / 2, it.pixmap().height() / 2)
                )
                self.scene().addItem(it)
                event.acceptProposedAction()
        except Exception:
            event.ignore()

    def keyPressEvent(self, event):
        """Handle arrow keys to nudge selected emblems.

        Arrow keys move by 1px; Shift+arrow moves by 5px.
        """
        key = event.key()
        step = 1
        try:
            if event.modifiers() & Qt.ShiftModifier:
                step = 5
        except Exception:
            pass

        dx = dy = 0
        if key == Qt.Key_Left:
            dx = -step
        elif key == Qt.Key_Right:
            dx = step
        elif key == Qt.Key_Up:
            dy = -step
        elif key == Qt.Key_Down:
            dy = step
        else:
            return super().keyPressEvent(event)

        moved = False
        try:
            for it in self.scene().selectedItems():
                # EmblemItem may not be defined yet at import time; detect by attribute
                if hasattr(it, "image_path"):
                    it.setPos(it.pos() + QPointF(dx, dy))
                    moved = True
        except Exception:
            moved = False

        if moved:
            event.accept()
        else:
            super().keyPressEvent(event)


APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ""  # leave empty; user picks
SETTINGS_FILE = APP_DIR / "settings.json"
CANVAS_W, CANVAS_H = 384, 256

# Helper: load image via PIL (works with pillow-dds plugin if installed)


def load_image(path: Path):
    try:
        img = Image.open(path)
        img = img.convert("RGBA")
        return img
    except Exception as e:
        # fallback: create placeholder
        img = Image.new("RGBA", (64, 64), (200, 50, 50, 255))
        return img


def pil2pixmap(img: Image.Image):
    data = img.tobytes("raw", "RGBA")
    qimg = QImage(data, img.width, img.height, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg)


# Simple replacement of mask colors in colored emblems
# Use the exact mask RGBs observed in the game's colored emblem assets
# (these are treated as opaque mask pixels; transparent areas remain transparent)
MASK_COLORS = [
    (0, 0, 129),  # blue -> color1 (observed)
    (0, 252, 129),  # green/teal -> color2 (observed)
    (247, 0, 129),  # pink -> color3 (observed)
]


def recolor_colored_emblem(img: Image.Image, color_vals):
    # color_vals: list of (r,g,b) or None, length up to 3
    px = img.convert("RGBA")
    data = px.getdata()
    out = []
    # Precompute mask vectors
    masks = MASK_COLORS
    # use euclidean distance threshold to allow for different mask RGB values (e.g., 129 blue)
    mask_thresh = 140.0
    mask_thresh_sq = mask_thresh * mask_thresh
    for p in data:
        r, g, b, a = p
        if a == 0:
            out.append((0, 0, 0, 0))
            continue
        # find nearest mask by squared distance
        best_idx = None
        best_ds = None
        for idx, mask in enumerate(masks):
            mr, mg, mb = mask
            ds = (r - mr) ** 2 + (g - mg) ** 2 + (b - mb) ** 2
            if best_ds is None or ds < best_ds:
                best_ds = ds
                best_idx = idx
        if best_ds is not None and best_ds <= mask_thresh_sq:
            idx = best_idx
            cv = (
                color_vals[idx]
                if idx < len(color_vals) and color_vals[idx] is not None
                else (r, g, b)
            )
            out.append((cv[0], cv[1], cv[2], a))
        else:
            out.append((r, g, b, a))
    res = Image.new("RGBA", px.size)
    res.putdata(out)
    return res


def recolor_pattern(
    img: Image.Image, color1_rgb, color2_rgb, color3_rgb=None, masks=None
):
    """Recolor pattern by detecting mask colors robustly.

    Strategy:
    - Gather distinct non-transparent colors and their counts.
    - For each distinct color, check distance to canonical masks (red, yellow, white).
      If close enough, map to the corresponding slot (color1/color2).
    - If no canonical masks are found, fall back to mapping the most frequent
      non-background colors to color1 and color2.
    This handles patterns that use slightly different RGB mask values.
    """
    px = img.convert("RGBA")
    data = list(px.getdata())
    # collect non-transparent color counts
    counts = {}
    for r, g, b, a in data:
        if a == 0:
            continue
        key = (r, g, b)
        counts[key] = counts.get(key, 0) + 1

    # canonical mask colors: red->color1, yellow->color2, white->color3
    red_mask = (255, 0, 0)
    yellow_mask = (255, 255, 0)
    white_mask = (255, 255, 255)

    # mapping from actual color -> target slot (1,2,3)
    mapping = {}

    def dist_sq(c1, c2):
        return (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2

    # thresholds (squared distances)
    close_thresh = 60**2

    # If masks are provided (pattern-specific exact colors), use exact matching
    if masks and any(masks):
        red_m, yellow_m, white_m = masks
        for col in counts:
            if red_m and col == red_m:
                mapping[col] = 1
            elif yellow_m and col == yellow_m:
                mapping[col] = 2
            elif white_m and col == white_m:
                mapping[col] = 3
    else:
        for col in counts:
            if dist_sq(col, red_mask) <= close_thresh:
                mapping[col] = 1
            elif dist_sq(col, yellow_mask) <= close_thresh:
                mapping[col] = 2
            elif dist_sq(col, white_mask) <= close_thresh:
                # white maps to color3 by convention
                mapping[col] = 3

    # If no canonical masks detected, pick top-2 frequent colors as color1/color2
    if not any(v in (1, 2) for v in mapping.values()):
        # sort colors by freq descending
        freq_sorted = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        if freq_sorted:
            # choose top two frequent colors and assign darker -> color1, lighter -> color2
            top_colors = [freq_sorted[i][0] for i in range(min(2, len(freq_sorted)))]
            if len(top_colors) == 1:
                mapping[top_colors[0]] = 1
            else:

                def luminance(c):
                    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

                c0, c1 = top_colors[0], top_colors[1]
                if luminance(c0) <= luminance(c1):
                    mapping[c0] = 1
                    mapping[c1] = 2
                else:
                    mapping[c1] = 1
                    mapping[c0] = 2

    out = []
    for r, g, b, a in data:
        if a == 0:
            out.append((0, 0, 0, 0))
            continue
        key = (r, g, b)
        slot = mapping.get(key)
        if slot == 1:
            cr, cg, cb = color1_rgb
            out.append((cr, cg, cb, a))
        elif slot == 2:
            cr, cg, cb = color2_rgb
            out.append((cr, cg, cb, a))
        elif slot == 3 and color3_rgb is not None:
            cr, cg, cb = color3_rgb
            out.append((cr, cg, cb, a))
        else:
            out.append((r, g, b, a))

    res = Image.new("RGBA", px.size)
    res.putdata(out)
    return res


def analyze_image_colors(img: Image.Image, name: str | None = None, top_n: int = 8):
    """Prints the most frequent non-transparent colors and distances to canonical masks."""
    px = img.convert("RGBA")
    data = list(px.getdata())
    counts = {}
    for r, g, b, a in data:
        if a == 0:
            continue
        counts[(r, g, b)] = counts.get((r, g, b), 0) + 1
    total = sum(counts.values())
    freq_sorted = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    # diagnostic output removed
    def dist(c1, c2):
        return math.sqrt(
            (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2
        )

    canonical = {
        "red_mask": (255, 0, 0),
        "yellow_mask": (255, 255, 0),
        "white": (255, 255, 255),
    }
    # diagnostic output removed


def detect_pattern_mask_colors(img: Image.Image):
    """Detect the exact RGB colors used in a pattern as mask colors for red/yellow/white.

    Returns a tuple (red_color, yellow_color, white_color) where each is an (r,g,b)
    triple or None if not found. This chooses the nearest actual pixel color to the
    canonical red/yellow/white values among non-transparent pixels.
    """
    px = img.convert("RGBA")
    data = list(px.getdata())
    counts = {}
    for r, g, b, a in data:
        if a == 0:
            continue
        counts[(r, g, b)] = counts.get((r, g, b), 0) + 1
    if not counts:
        return (None, None, None)

    def nearest_color(target):
        best = None
        bestd = None
        for col in counts:
            d = (
                (col[0] - target[0]) ** 2
                + (col[1] - target[1]) ** 2
                + (col[2] - target[2]) ** 2
            )
            if bestd is None or d < bestd:
                bestd = d
                best = col
        return best

    red = nearest_color((255, 0, 0))
    yellow = nearest_color((255, 255, 0))
    white = nearest_color((255, 255, 255))
    return (red, yellow, white)


def infer_emblem_mask_colors(img: Image.Image):
    """Infer the actual RGB values used in a colored emblem for mask slots.

    Returns a list of length 3 with (r,g,b) tuples or None for each slot.
    Uses nearest-neighbor to MASK_COLORS on the most frequent colors in the image.
    """
    px = img.convert("RGBA")
    data = list(px.getdata())
    counts = {}
    for r, g, b, a in data:
        if a == 0:
            continue
        counts[(r, g, b)] = counts.get((r, g, b), 0) + 1
    if not counts:
        return [None, None, None]

    # map actual color -> nearest mask index
    mapping = {}
    for col in counts:
        best_idx = None
        best_ds = None
        for idx, mask in enumerate(MASK_COLORS):
            ds = (
                (col[0] - mask[0]) ** 2
                + (col[1] - mask[1]) ** 2
                + (col[2] - mask[2]) ** 2
            )
            if best_ds is None or ds < best_ds:
                best_ds = ds
                best_idx = idx
        # only accept mapping if reasonably close
        mapping.setdefault(best_idx, []).append((col, counts[col]))

    # for each slot pick the most frequent actual color
    result = [None, None, None]
    for idx in range(3):
        lst = mapping.get(idx)
        if not lst:
            continue
        lst_sorted = sorted(lst, key=lambda x: x[1], reverse=True)
        result[idx] = lst_sorted[0][0]
    return result


class EmblemItem(QGraphicsPixmapItem):
    def __init__(
        self, image_path: Path, pil_image: Image.Image, texture_type: str = "colored"
    ):
        super().__init__()
        self.setFlags(
            QGraphicsPixmapItem.ItemIsMovable
            | QGraphicsPixmapItem.ItemIsSelectable
            | QGraphicsPixmapItem.ItemIsFocusable
        )
        self.image_path = Path(image_path)
        self.base_image = pil_image  # original PIL image
        self.texture_type = texture_type
        # support independent X/Y scaling
        self.scale_x = 1.0
        self.scale_y = 1.0
        # mirroring flags
        self.mirror_x = False
        self.mirror_y = False
        # per-emblem color values as (r,g,b) 0-255 or None; index 0..2
        self.color_vals = [None, None, None]
        self.update_pixmap()

    def update_pixmap(self, color_vals=None):
        # legacy: if external_pixmap supplied via caller, that will be set directly
        img = self.base_image
        cvs = color_vals if color_vals is not None else self.color_vals
        if self.texture_type == "colored" and any(c is not None for c in cvs):
            img = recolor_colored_emblem(self.base_image, cvs)
        pix = pil2pixmap(img)
        if self.scale_x != 1.0 or self.scale_y != 1.0:
            pix = pix.scaled(
                int(pix.width() * self.scale_x),
                int(pix.height() * self.scale_y),
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation,
            )
        self.setPixmap(pix)

    def set_color(self, idx: int, qcolor: QColor | None):
        """Set a single color index (0..2) from a QColor or None and update."""
        if idx < 0 or idx > 2:
            return
        if qcolor is None:
            self.color_vals[idx] = None
        else:
            self.color_vals[idx] = (qcolor.red(), qcolor.green(), qcolor.blue())
        # pixmap update is performed by the MainWindow to allow pattern masking
        return

    def set_scale_factor(self, s: float):
        # uniform setter: apply to both axes
        if s is None:
            return
        try:
            s = float(s)
        except Exception:
            return
        if s < 0.1:
            s = 0.1
        if s > 1.0:
            s = 1.0
        self.scale_x = s
        self.scale_y = s
        # pixmap update is performed by the MainWindow to allow pattern masking
        return

    def set_scale_x(self, s: float):
        if s is None:
            return
        try:
            s = float(s)
        except Exception:
            return
        if s < 0.1:
            s = 0.1
        if s > 1.0:
            s = 1.0
        self.scale_x = s
        return

    def set_scale_y(self, s: float):
        if s is None:
            return
        try:
            s = float(s)
        except Exception:
            return
        if s < 0.1:
            s = 0.1
        if s > 1.0:
            s = 1.0
        self.scale_y = s
        return


class SettingsWindow(QWidget):
    """Persistent settings window to choose base game and mod folders.

    Shows current paths in read-only line edits with buttons to change them.
    """

    def __init__(self, parent: "MainWindow"):
        super().__init__(parent)
        # make this widget a top-level window (not embedded child)
        try:
            self.setWindowFlags(self.windowFlags() | Qt.Window)
        except Exception:
            pass
        self.parent = parent
        self.setWindowTitle("Settings")
        self.setMinimumWidth(600)
        layout = QVBoxLayout(self)

        # Base game row
        row_base = QHBoxLayout()
        lbl_base = QLabel("Base game folder:")
        self.base_le = QLineEdit()
        self.base_le.setReadOnly(True)
        btn_base = QPushButton("Choose...")
        btn_base.clicked.connect(self.choose_base)
        row_base.addWidget(lbl_base)
        row_base.addWidget(self.base_le, 1)
        row_base.addWidget(btn_base)
        layout.addLayout(row_base)

        # Mod folder row
        row_mod = QHBoxLayout()
        lbl_mod = QLabel("Mod folder (optional):")
        self.mod_le = QLineEdit()
        self.mod_le.setReadOnly(True)
        btn_mod = QPushButton("Choose...")
        btn_mod.clicked.connect(self.choose_mod)
        row_mod.addWidget(lbl_mod)
        row_mod.addWidget(self.mod_le, 1)
        row_mod.addWidget(btn_mod)
        layout.addLayout(row_mod)

        # Actions
        actions = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_and_apply)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        actions.addStretch()
        actions.addWidget(save_btn)
        actions.addWidget(close_btn)
        layout.addLayout(actions)

        self.refresh()

    def refresh(self):
        s = self.parent.settings
        self.base_le.setText(s.get("base_game", "") or "")
        self.mod_le.setText(s.get("mod_folder", "") or "")

    def choose_base(self):
        dlg = QFileDialog(self, "Select base game folder")
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec():
            self.base_le.setText(dlg.selectedFiles()[0])

    def choose_mod(self):
        dlg = QFileDialog(self, "Select mod folder")
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec():
            self.mod_le.setText(dlg.selectedFiles()[0])

    def save_and_apply(self):
        # write to parent settings and re-scan assets
        self.parent.settings["base_game"] = self.base_le.text()
        self.parent.settings["mod_folder"] = self.mod_le.text()
        self.parent.save_settings()
        try:
            self.parent.scan_assets()
        except Exception:
            pass
        self.close()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EU5 Flag Editor")
        self.resize(1000, 600)
        self.settings = self.load_settings()
        self.assets = {"patterns": {}, "colored_emblems": {}, "textured_emblems": {}}
        # per-texture saved emblem colors: name -> [c1,c2,c3] tuples or None
        self.emblem_saved_colors = {}
        self.setup_ui()
        if self.settings.get("base_game"):
            self.scan_assets()

    def load_settings(self):
        if SETTINGS_FILE.exists():
            return json.loads(SETTINGS_FILE.read_text())
        return {"base_game": "", "mod_folder": ""}

    def save_settings(self):
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(self.settings, indent=2))

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        h = QHBoxLayout(central)

        left = QVBoxLayout()
        self.patterns_list = QListWidget()
        self.patterns_list.setSelectionMode(QAbstractItemView.SingleSelection)
        # search box for patterns
        self.patterns_search = QLineEdit()
        self.patterns_search.setPlaceholderText("Search patterns...")
        self.patterns_search.textChanged.connect(self.filter_patterns)
        self.emblems_list = DraggableListWidget(asset_type="colored_emblems")
        self.emblems_list.setDragEnabled(True)
        left.addWidget(QLabel("Patterns (patterns)"))
        left.addWidget(self.patterns_list)
        left.addWidget(self.patterns_search)
        left.addWidget(QLabel("Colored Emblems (colored_emblems)"))
        # search box for emblems
        self.emblems_search = QLineEdit()
        self.emblems_search.setPlaceholderText("Search emblems...")
        self.emblems_search.textChanged.connect(self.filter_emblems)
        left.addWidget(self.emblems_list)
        # textured emblems list (hidden by default)
        self.textured_emblems_list = DraggableListWidget(asset_type="textured_emblems")
        self.textured_emblems_list.setDragEnabled(True)
        self.textured_emblems_list.setVisible(False)
        left.addWidget(self.textured_emblems_list)
        left.addWidget(self.emblems_search)

        # toggle button to switch between colored and textured emblem views
        self.toggle_emblem_btn = QPushButton("Show Textured Emblems")
        self.toggle_emblem_btn.setCheckable(True)
        self.toggle_emblem_btn.clicked.connect(self.toggle_emblem_list)
        left.addWidget(self.toggle_emblem_btn)

        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self.open_settings)
        left.addWidget(settings_btn)

        h.addLayout(left, 1)

        # Center canvas
        self.scene = QGraphicsScene(0, 0, CANVAS_W, CANVAS_H)
        self.view = DropGraphicsView(self.scene, owner=self)
        self.view.setFixedSize(CANVAS_W + 10, CANVAS_H + 10)
        self.view.setDragMode(QGraphicsView.RubberBandDrag)
        h.addWidget(self.view, 0)

        # Right panel controls
        right = QVBoxLayout()
        form = QFormLayout()
        # pattern colour pickers (default values from named colors file if available)
        self.col1_btn = QPushButton("Pattern Colour 1")
        self.col2_btn = QPushButton("Pattern Colour 2")
        self.col3_btn = QPushButton("Pattern Colour 3")
        self.col1_label = QLabel("Pattern Colour 1:")
        self.col2_label = QLabel("Pattern Colour 2:")
        self.col3_label = QLabel("Pattern Colour 3:")
        self.col1_btn.clicked.connect(self.pick_color1)
        self.col2_btn.clicked.connect(self.pick_color2)
        self.col3_btn.clicked.connect(self.pick_color3)
        form.addRow(self.col1_label, self.col1_btn)
        form.addRow(self.col2_label, self.col2_btn)
        form.addRow(self.col3_label, self.col3_btn)

        # emblem-specific colour pickers (for selected emblem). first two correspond to the pattern colours
        self.em_col1_btn = QPushButton("Emblem Colour 1")
        self.em_col2_btn = QPushButton("Emblem Colour 2")
        self.em_col3_btn = QPushButton("Emblem Colour 3")
        self.em_col1_label = QLabel("Emblem Colour 1:")
        self.em_col2_label = QLabel("Emblem Colour 2:")
        self.em_col3_label = QLabel("Emblem Colour 3:")
        self.em_col1_btn.clicked.connect(self.pick_emblem_color1)
        self.em_col2_btn.clicked.connect(self.pick_emblem_color2)
        self.em_col3_btn.clicked.connect(self.pick_emblem_color3)
        form.addRow(self.em_col1_label, self.em_col1_btn)
        form.addRow(self.em_col2_label, self.em_col2_btn)
        form.addRow(self.em_col3_label, self.em_col3_btn)

        right.addLayout(form)

        # color is chosen via color picker buttons above (RGB only)

        # Apply button removed; emblem pickers update immediately

        scale_label = QLabel("Scale (selected)")
        # separate X and Y scale controls (percent)
        # Scale X
        sx_label = QLabel("Scale X (selected)")
        self.scale_x_slider = QSlider(Qt.Horizontal)
        self.scale_x_slider.setRange(10, 100)
        self.scale_x_slider.setValue(100)
        right.addWidget(sx_label)
        right.addWidget(self.scale_x_slider)
        sx_row = QHBoxLayout()
        sx_btn25 = QPushButton("25%")
        sx_btn50 = QPushButton("50%")
        sx_btn100 = QPushButton("100%")
        self.scale_x_spin = QSpinBox()
        self.scale_x_spin.setRange(10, 100)
        self.scale_x_spin.setValue(100)
        self.scale_x_spin.setSuffix("%")
        self.scale_x_spin.setFixedWidth(80)
        sx_btn25.clicked.connect(lambda: self.scale_x_spin.setValue(25))
        sx_btn50.clicked.connect(lambda: self.scale_x_spin.setValue(50))
        sx_btn100.clicked.connect(lambda: self.scale_x_spin.setValue(100))
        sx_row.addWidget(sx_btn25)
        sx_row.addWidget(sx_btn50)
        sx_row.addWidget(sx_btn100)
        sx_row.addStretch()
        sx_row.addWidget(self.scale_x_spin)
        right.addLayout(sx_row)

        # Scale Y
        sy_label = QLabel("Scale Y (selected)")
        self.scale_y_slider = QSlider(Qt.Horizontal)
        self.scale_y_slider.setRange(10, 100)
        self.scale_y_slider.setValue(100)
        right.addWidget(sy_label)
        right.addWidget(self.scale_y_slider)
        sy_row = QHBoxLayout()
        sy_btn25 = QPushButton("25%")
        sy_btn50 = QPushButton("50%")
        sy_btn100 = QPushButton("100%")
        self.scale_y_spin = QSpinBox()
        self.scale_y_spin.setRange(10, 100)
        self.scale_y_spin.setValue(100)
        self.scale_y_spin.setSuffix("%")
        self.scale_y_spin.setFixedWidth(80)
        sy_btn25.clicked.connect(lambda: self.scale_y_spin.setValue(25))
        sy_btn50.clicked.connect(lambda: self.scale_y_spin.setValue(50))
        sy_btn100.clicked.connect(lambda: self.scale_y_spin.setValue(100))
        sy_row.addWidget(sy_btn25)
        sy_row.addWidget(sy_btn50)
        sy_row.addWidget(sy_btn100)
        sy_row.addStretch()
        sy_row.addWidget(self.scale_y_spin)
        right.addLayout(sy_row)

        # Mirror controls
        mirror_row = QHBoxLayout()
        self.mirror_x_btn = QPushButton("Mirror X")
        self.mirror_x_btn.setCheckable(True)
        self.mirror_y_btn = QPushButton("Mirror Y")
        self.mirror_y_btn.setCheckable(True)
        self.mirror_x_btn.clicked.connect(self.mirror_x_toggled)
        self.mirror_y_btn.clicked.connect(self.mirror_y_toggled)
        mirror_row.addWidget(self.mirror_x_btn)
        mirror_row.addWidget(self.mirror_y_btn)
        mirror_row.addStretch()
        right.addLayout(mirror_row)

        # wire up synchronization: sliders <-> spinboxes and handlers
        self.scale_x_spin.valueChanged.connect(lambda v: self.scale_x_slider.setValue(v))
        self.scale_x_slider.valueChanged.connect(lambda v: (self.scale_x_spin.blockSignals(True), self.scale_x_spin.setValue(v), self.scale_x_spin.blockSignals(False), self.scale_x_changed(v)))
        self.scale_y_spin.valueChanged.connect(lambda v: self.scale_y_slider.setValue(v))
        self.scale_y_slider.valueChanged.connect(lambda v: (self.scale_y_spin.blockSignals(True), self.scale_y_spin.setValue(v), self.scale_y_spin.blockSignals(False), self.scale_y_changed(v)))

        # center controls: center both axes or individually
        ctr_row = QHBoxLayout()
        center_both_btn = QPushButton("Center Both")
        center_both_btn.setToolTip("Center the selected emblem on both X and Y axes")
        center_both_btn.clicked.connect(lambda: self.center_selected("both"))
        center_x_btn = QPushButton("Center X")
        center_x_btn.setToolTip("Center the selected emblem on the X axis")
        center_x_btn.clicked.connect(lambda: self.center_selected("x"))
        center_y_btn = QPushButton("Center Y")
        center_y_btn.setToolTip("Center the selected emblem on the Y axis")
        center_y_btn.clicked.connect(lambda: self.center_selected("y"))
        ctr_row.addWidget(center_both_btn)
        ctr_row.addWidget(center_x_btn)
        ctr_row.addWidget(center_y_btn)
        right.addLayout(ctr_row)

        delete_btn = QPushButton("Remove Selected")
        delete_btn.setToolTip("Remove the selected emblem from the canvas")
        delete_btn.clicked.connect(self.delete_selected)
        right.addWidget(delete_btn)

        export_btn = QPushButton("Export CoA Block")
        export_btn.clicked.connect(self.export_coa)
        right.addWidget(export_btn)

        h.addLayout(right, 1)

        # connect list interactions
        self.patterns_list.itemDoubleClicked.connect(self.select_pattern)
        # respond to selection changes in the scene so controls reflect selected emblem
        self.scene.selectionChanged.connect(self.on_selection_changed)
        # double-click or context action to add emblem to scene as a fallback
        # colored list double-click
        self.emblems_list.itemDoubleClicked.connect(
            partial(self._double_click_add_from_list, self.emblems_list)
        )
        # textured list double-click
        self.textured_emblems_list.itemDoubleClicked.connect(
            partial(self._double_click_add_from_list, self.textured_emblems_list)
        )

        # show loaded pattern as background when selected
        self.current_pattern_item = None
        self.current_pattern_name = None
        # default: hide extra colour controls until a pattern/emblem is selected
        try:
            self.update_pattern_color_controls(1)
            self.update_emblem_color_controls(1)
        except Exception:
            pass

    def scan_assets(self):
        # Treat empty strings as no-folder; prefer mod files over base files
        base_str = self.settings.get("base_game", "")
        base = Path(base_str) if base_str else None
        mod_str = self.settings.get("mod_folder", "")
        mod = Path(mod_str) if mod_str else None

        def gather(subpath):
            results = {}

            def scan_root(root_path):
                # Check both the exact subpath and, if subpath starts with 'game',
                # the subpath without the leading 'game' component (mods often omit it).
                found = {}
                p = root_path / subpath
                if p.exists():
                    for f in p.iterdir():
                        if f.suffix.lower() in [".dds", ".png", ".tga"]:
                            found[f.name] = f.resolve()
                # try alternative location: drop leading 'game' if present
                parts = subpath.parts
                if parts and parts[0] == "game":
                    alt = root_path.joinpath(*parts[1:])
                    if alt.exists():
                        for f in alt.iterdir():
                            if f.suffix.lower() in [".dds", ".png", ".tga"]:
                                found[f.name] = f.resolve()
                return found

            if base and base.exists():
                results.update(scan_root(base))
            if mod and mod.exists():
                # mod overrides base; update after base
                results.update(scan_root(mod))
            return results

        self.assets["patterns"] = gather(
            Path("game") / "main_menu" / "gfx" / "coat_of_arms" / "patterns"
        )
        self.assets["colored_emblems"] = gather(
            Path("game") / "main_menu" / "gfx" / "coat_of_arms" / "colored_emblems"
        )
        self.assets["textured_emblems"] = gather(
            Path("game") / "main_menu" / "gfx" / "coat_of_arms" / "textured_emblems"
        )
        self.populate_lists()
        # load named colors file if available
        named_colors_path = None
        if base:
            p = (
                base
                / Path("game")
                / "main_menu"
                / "common"
                / "named_colors"
                / "01_coa.txt"
            )
            if p.exists():
                named_colors_path = p
        # do not use hardcoded paths; only use provided base/mod folders
        # if named colors not found in base, leave as empty and proceed
        self.named_colors = {}
        if named_colors_path:
            try:
                self.named_colors = self.parse_named_colors(named_colors_path)
            except Exception:
                self.named_colors = {}

        # initialize color pickers with defaults from named colors
        # prefer 'white' and 'yellow_mid' if available
        def set_default_btn(btn, qname):
            if self.named_colors.get(qname):
                r, g, b = self.named_colors[qname]
                qc = QColor(int(r * 255), int(g * 255), int(b * 255))
            else:
                qc = QColor(255, 255, 255)
            btn._qcolor = qc
            btn.setStyleSheet(f"background-color: {qc.name()};")

        set_default_btn(self.col1_btn, "white")
        set_default_btn(self.col2_btn, "yellow_mid")
        set_default_btn(self.col3_btn, "white")

        # set emblem picker defaults to the known emblem mask colours
        try:
            for idx, b in enumerate(
                (self.em_col1_btn, self.em_col2_btn, self.em_col3_btn)
            ):
                mc = MASK_COLORS[idx]
                qc = QColor(mc[0], mc[1], mc[2])
                b._qcolor = qc
                b.setStyleSheet(f"background-color: {qc.name()};")
        except Exception:
            pass

        # auto-select a sensible default pattern if present
        default_pattern = "pattern_solid.dds"
        if default_pattern in self.assets.get("patterns", {}):
            # find the matching item in the list and select it
            for i in range(self.patterns_list.count()):
                it = self.patterns_list.item(i)
                if it and it.text() == default_pattern:
                    self.patterns_list.setCurrentItem(it)
                    try:
                        self.select_pattern(it)
                    except Exception:
                        pass
                    break

    def pick_color1(self):
        init = getattr(self.col1_btn, "_qcolor", QColor(255, 255, 255))
        c = QColorDialog.getColor(
            init, self, "Pick Pattern Colour 1", QColorDialog.ShowAlphaChannel
        )
        if c.isValid():
            self.col1_btn._qcolor = c
            self.col1_btn.setStyleSheet(f"background-color: {c.name()};")
            # update pattern display if any
            self._update_pattern_display()

    def pick_color2(self):
        init = getattr(self.col2_btn, "_qcolor", QColor(255, 255, 255))
        c = QColorDialog.getColor(
            init, self, "Pick Pattern Colour 2", QColorDialog.ShowAlphaChannel
        )
        if c.isValid():
            self.col2_btn._qcolor = c
            self.col2_btn.setStyleSheet(f"background-color: {c.name()};")
            # update pattern display if any
            self._update_pattern_display()

    def pick_color3(self):
        init = getattr(self.col3_btn, "_qcolor", QColor(255, 255, 255))
        c = QColorDialog.getColor(
            init, self, "Pick Pattern Colour 3", QColorDialog.ShowAlphaChannel
        )
        if c.isValid():
            self.col3_btn._qcolor = c
            self.col3_btn.setStyleSheet(f"background-color: {c.name()};")
            # update pattern display if any
            self._update_pattern_display()

    def pick_emblem_color1(self):
        sel = [i for i in self.scene.selectedItems() if isinstance(i, EmblemItem)]
        if not sel:
            QMessageBox.information(self, "Color", "Select an emblem first")
            return
        item = sel[0]
        if getattr(item, "texture_type", "colored") != "colored":
            QMessageBox.information(self, "Color", "Textured emblems have no colors")
            return
        # initial color: emblem color1 if present, else emblem fallback to global color1
        init = getattr(self.em_col1_btn, "_qcolor", None)
        if init is None:
            # try emblem current color
            cv = item.color_vals[0]
            if cv:
                init = QColor(int(cv[0]), int(cv[1]), int(cv[2]))
            else:
                init = getattr(self.col1_btn, "_qcolor", QColor(255, 255, 255))
        c = QColorDialog.getColor(
            init, self, "Pick Emblem Colour 1", QColorDialog.ShowAlphaChannel
        )
        if c.isValid():
            self.em_col1_btn._qcolor = c
            self.em_col1_btn.setStyleSheet(f"background-color: {c.name()};")
            item.set_color(0, c)
            self.update_emblem_pixmap(item)
            # save this choice for this emblem texture
            try:
                self.emblem_saved_colors[item.image_path.name] = list(item.color_vals)
            except Exception:
                pass

    def pick_emblem_color2(self):
        sel = [i for i in self.scene.selectedItems() if isinstance(i, EmblemItem)]
        if not sel:
            QMessageBox.information(self, "Color", "Select an emblem first")
            return
        item = sel[0]
        if getattr(item, "texture_type", "colored") != "colored":
            QMessageBox.information(self, "Color", "Textured emblems have no colors")
            return
        init = getattr(self.em_col2_btn, "_qcolor", None)
        if init is None:
            cv = item.color_vals[1]
            if cv:
                init = QColor(int(cv[0]), int(cv[1]), int(cv[2]))
            else:
                init = getattr(self.col2_btn, "_qcolor", QColor(255, 255, 255))
        c = QColorDialog.getColor(
            init, self, "Pick Emblem Colour 2", QColorDialog.ShowAlphaChannel
        )
        if c.isValid():
            self.em_col2_btn._qcolor = c
            self.em_col2_btn.setStyleSheet(f"background-color: {c.name()};")
            item.set_color(1, c)
            self.update_emblem_pixmap(item)
            try:
                self.emblem_saved_colors[item.image_path.name] = list(item.color_vals)
            except Exception:
                pass

    def pick_emblem_color3(self):
        sel = [i for i in self.scene.selectedItems() if isinstance(i, EmblemItem)]
        if not sel:
            QMessageBox.information(self, "Color", "Select an emblem first")
            return
        item = sel[0]
        if getattr(item, "texture_type", "colored") != "colored":
            QMessageBox.information(self, "Color", "Textured emblems have no colors")
            return
        init = getattr(self.em_col3_btn, "_qcolor", None)
        if init is None:
            cv = item.color_vals[2]
            if cv:
                init = QColor(int(cv[0]), int(cv[1]), int(cv[2]))
            else:
                init = QColor(255, 255, 255)
        c = QColorDialog.getColor(
            init, self, "Pick Emblem Colour 3", QColorDialog.ShowAlphaChannel
        )
        if c.isValid():
            self.em_col3_btn._qcolor = c
            self.em_col3_btn.setStyleSheet(f"background-color: {c.name()};")
            item.set_color(2, c)
            try:
                self.emblem_saved_colors[item.image_path.name] = list(item.color_vals)
            except Exception:
                pass
            self.update_emblem_pixmap(item)

    def populate_lists(self):
        self.patterns_list.clear()
        for name, path in sorted(self.assets["patterns"].items()):
            item = QListWidgetItem(name)
            # try to make thumbnail
            img = load_image(path)
            thumb = pil2pixmap(img.resize((96, int(96 * img.height / img.width))))
            item.setIcon(thumb)
            self.patterns_list.addItem(item)
        self.emblems_list.clear()
        for name, path in sorted(self.assets["colored_emblems"].items()):
            item = QListWidgetItem(name)
            img = load_image(path)
            thumb = pil2pixmap(img.resize((64, int(64 * img.height / img.width))))
            item.setIcon(thumb)
            self.emblems_list.addItem(item)
        # textured emblems
        self.textured_emblems_list.clear()
        for name, path in sorted(self.assets.get("textured_emblems", {}).items()):
            item = QListWidgetItem(name)
            img = load_image(path)
            thumb = pil2pixmap(img.resize((64, int(64 * img.height / img.width))))
            item.setIcon(thumb)
            self.textured_emblems_list.addItem(item)
        # clear any search filters
        try:
            self.patterns_search.setText("")
        except Exception:
            pass
        try:
            self.emblems_search.setText("")
        except Exception:
            pass

    def filter_patterns(self, text: str):
        t = (text or "").lower()
        for i in range(self.patterns_list.count()):
            it = self.patterns_list.item(i)
            if not it:
                continue
            it.setHidden(t not in it.text().lower())

    def filter_emblems(self, text: str):
        t = (text or "").lower()
        for lst in (self.emblems_list, self.textured_emblems_list):
            for i in range(lst.count()):
                it = lst.item(i)
                if not it:
                    continue
                it.setHidden(t not in it.text().lower())

    def open_settings(self):
        # open persistent settings window
        try:
            if (
                getattr(self, "settings_window", None)
                and self.settings_window.isVisible()
            ):
                self.settings_window.raise_()
                return
        except Exception:
            pass
        self.settings_window = SettingsWindow(self)
        self.settings_window.show()

    def _double_click_add_from_list(self, lst: DraggableListWidget, item: QListWidgetItem):
        # Helper for double-click: add emblem from specified list
        name = item.text()
        path = self.assets.get(lst.asset_type, {}).get(name)
        if not path:
            return
        typ = "textured" if lst.asset_type == "textured_emblems" else "colored"
        self.add_emblem_to_scene(Path(path), typ)

    def toggle_emblem_list(self):
        # toggle between colored and textured emblem lists
        show_textured = self.toggle_emblem_btn.isChecked()
        if show_textured:
            self.emblems_list.setVisible(False)
            self.textured_emblems_list.setVisible(True)
            self.toggle_emblem_btn.setText("Show Colored Emblems")
        else:
            self.emblems_list.setVisible(True)
            self.textured_emblems_list.setVisible(False)
            self.toggle_emblem_btn.setText("Show Textured Emblems")

    def select_pattern(self, item: QListWidgetItem):
        name = item.text()
        path = self.assets["patterns"].get(name)
        if not path:
            return
        img = load_image(path)
        # detect exact mask colors for this pattern (red/yellow/white)
        try:
            masks = detect_pattern_mask_colors(img)
            self._current_pattern_masks = masks
        except Exception:
            self._current_pattern_masks = (None, None, None)
        # set the pattern colour pickers to the detected mask colours (exact)
        try:
            if self._current_pattern_masks and any(self._current_pattern_masks):
                m1, m2, m3 = self._current_pattern_masks
                if m1:
                    qc = QColor(m1[0], m1[1], m1[2])
                    self.col1_btn._qcolor = qc
                    self.col1_btn.setStyleSheet(f"background-color: {qc.name()};")
                if m2:
                    qc = QColor(m2[0], m2[1], m2[2])
                    self.col2_btn._qcolor = qc
                    self.col2_btn.setStyleSheet(f"background-color: {qc.name()};")
                if m3:
                    qc = QColor(m3[0], m3[1], m3[2])
                    self.col3_btn._qcolor = qc
                    self.col3_btn.setStyleSheet(f"background-color: {qc.name()};")
        except Exception:
            pass
        # recolor according to current pickers
        qc1 = getattr(self.col1_btn, "_qcolor", QColor(255, 255, 255))
        qc1 = getattr(self.col1_btn, "_qcolor", QColor(255, 255, 255))
        qc2 = getattr(self.col2_btn, "_qcolor", QColor(255, 255, 255))
        qc3 = getattr(self.col3_btn, "_qcolor", QColor(255, 255, 255))
        c1 = (qc1.red(), qc1.green(), qc1.blue())
        c2 = (qc2.red(), qc2.green(), qc2.blue())
        c3 = (qc3.red(), qc3.green(), qc3.blue())
        pil = recolor_pattern(
            img, c1, c2, c3, masks=getattr(self, "_current_pattern_masks", None)
        )
        pix = pil2pixmap(pil)
        # remove old background
        if self.current_pattern_item:
            self.scene.removeItem(self.current_pattern_item)
        bg = QGraphicsPixmapItem(pix)
        # fill the canvas exactly and keep pattern locked in the very back
        bg.setPos(0, 0)
        bg.setZValue(-1000)
        try:
            # make non-interactive and non-selectable
            bg.setFlag(QGraphicsItem.ItemIsSelectable, False)
            bg.setFlag(QGraphicsItem.ItemIsMovable, False)
        except Exception:
            pass
        bg.setAcceptHoverEvents(False)
        bg.setAcceptedMouseButtons(Qt.NoButton)
        self.scene.addItem(bg)
        self.current_pattern_item = bg
        # remember selected pattern filename for export
        self.current_pattern_name = name

        # store original pattern pil for later recolor updates
        self._current_pattern_pil = img
        # determine how many colour slots this pattern uses and update controls
        try:
            slots = self._determine_pattern_slots(img, getattr(self, "_current_pattern_masks", None))
        except Exception:
            slots = 1
        self.update_pattern_color_controls(slots)

    def _update_pattern_display(self):
        if not getattr(self, "current_pattern_item", None) or not getattr(
            self, "_current_pattern_pil", None
        ):
            return
        qc1 = getattr(self.col1_btn, "_qcolor", QColor(255, 255, 255))
        qc2 = getattr(self.col2_btn, "_qcolor", QColor(255, 255, 255))
        c1 = (qc1.red(), qc1.green(), qc1.blue())
        c2 = (qc2.red(), qc2.green(), qc2.blue())
        qc3 = getattr(self.col3_btn, "_qcolor", QColor(255, 255, 255))
        c3 = (qc3.red(), qc3.green(), qc3.blue())
        pil = recolor_pattern(
            self._current_pattern_pil,
            c1,
            c2,
            c3,
            masks=getattr(self, "_current_pattern_masks", None),
        )
        pil = pil.resize((CANVAS_W, CANVAS_H))
        pix = pil2pixmap(pil)
        self.current_pattern_item.setPixmap(pix)

    def _determine_pattern_slots(self, img: Image.Image, masks=None) -> int:
        # return 1/2/3 depending on how many color slots the pattern uses
        # try mask-based detection first
        try:
            # Build color frequency map
            px = img.convert("RGBA")
            counts = {}
            total = 0
            for r, g, b, a in px.getdata():
                if a == 0:
                    continue
                counts[(r, g, b)] = counts.get((r, g, b), 0) + 1
                total += 1

            # helper: nearest actual color to a target
            def nearest_and_count(target):
                best = None
                bestd = None
                for col, cnt in counts.items():
                    d = (col[0] - target[0]) ** 2 + (col[1] - target[1]) ** 2 + (col[2] - target[2]) ** 2
                    if bestd is None or d < bestd:
                        bestd = d
                        best = (col, cnt, d)
                return best  # (col, count, dist_sq) or None

            # check canonical masks red/yellow/white
            slots_found = set()
            thresh_sq = (90) ** 2
            rc = nearest_and_count((255, 0, 0))
            yc = nearest_and_count((255, 255, 0))
            wc = nearest_and_count((255, 255, 255))
            # require both proximity and non-trivial pixel count to consider slot present
            if rc and rc[2] <= thresh_sq and rc[1] >= max(8, total * 0.001):
                slots_found.add(1)
            if yc and yc[2] <= thresh_sq and yc[1] >= max(8, total * 0.001):
                slots_found.add(2)
            if wc and wc[2] <= thresh_sq and wc[1] >= max(8, total * 0.001):
                slots_found.add(3)

            # if no clear canonical masks detected, fall back to number of distinct colors
            if not slots_found:
                uniq = len(counts)
                if uniq >= 3:
                    return 3
                if uniq == 2:
                    return 2
                return 1
            return max(slots_found)
        except Exception:
            return 1

    def update_pattern_color_controls(self, slots: int):
        # ensure slot is 1..3
        if slots < 1:
            slots = 1
        if slots > 3:
            slots = 3
        # slot 1 always visible
        self.col1_label.setVisible(True)
        self.col1_btn.setVisible(True)
        # slot 2
        show2 = slots >= 2
        self.col2_label.setVisible(show2)
        self.col2_btn.setVisible(show2)
        # slot 3
        show3 = slots >= 3
        self.col3_label.setVisible(show3)
        self.col3_btn.setVisible(show3)

    def update_emblem_color_controls(self, slots: int):
        # slots: number of emblem colour slots (1..3). Show/hide accordingly.
        # allow slots==0 to hide all emblem colour controls for textured emblems
        if slots <= 0:
            # hide all emblem controls
            self.em_col1_label.setVisible(False)
            self.em_col1_btn.setVisible(False)
            self.em_col2_label.setVisible(False)
            self.em_col2_btn.setVisible(False)
            self.em_col3_label.setVisible(False)
            self.em_col3_btn.setVisible(False)
            return
        if slots < 1:
            slots = 1
        if slots > 3:
            slots = 3
        self.em_col1_label.setVisible(True)
        self.em_col1_btn.setVisible(True)
        show2 = slots >= 2
        self.em_col2_label.setVisible(show2)
        self.em_col2_btn.setVisible(show2)
        show3 = slots >= 3
        self.em_col3_label.setVisible(show3)
        self.em_col3_btn.setVisible(show3)
        # update all emblems so they are masked by the new pattern
        for it in [i for i in self.scene.items() if isinstance(i, EmblemItem)]:
            self.update_emblem_pixmap(it)

    def add_emblem_to_scene(
        self, path: Path, typ: str = "colored", scene_pos: QPointF = None
    ):
        pil = load_image(path)
        # debug: analyze colors for african assets to help tune recolour thresholds
        try:
            if "african" in path.name.lower():
                analyze_image_colors(pil, path.name)
        except Exception:
            pass
        it = EmblemItem(path, pil, texture_type=typ)
        # textured emblems do not have colors: leave color_vals as all None
        if it.texture_type == "colored":
            # if there are saved colors for this texture, apply them to the item
            try:
                saved = self.emblem_saved_colors.get(it.image_path.name)
                if saved:
                    # ensure list length 3
                    it.color_vals = list(saved[:3]) + [None] * (3 - len(saved))
                else:
                    # no saved colors: infer which mask slots are actually present in the emblem
                    inferred = infer_emblem_mask_colors(it.base_image)
                    new_vals = [None, None, None]
                    if inferred:
                        for idx in range(min(3, len(inferred))):
                            if inferred[idx]:
                                new_vals[idx] = (
                                    inferred[idx][0], inferred[idx][1], inferred[idx][2]
                                )
                    it.color_vals = new_vals
            except Exception:
                # fallback: leave color_vals as None
                pass
        if scene_pos is None:
            # place at center
            scene_pos = QPointF(CANVAS_W / 2, CANVAS_H / 2)
        it.setPos(
            scene_pos - QPointF(it.pixmap().width() / 2, it.pixmap().height() / 2)
        )
        # place emblem above existing emblems and above the pattern
        # compute highest z among existing EmblemItems
        existing_emblems = [i for i in self.scene.items() if isinstance(i, EmblemItem)]
        max_z = max((i.zValue() for i in existing_emblems), default=-999)
        it.setZValue(max_z + 1)
        self.scene.addItem(it)
        # render emblem pixmap with current pattern masking
        self.update_emblem_pixmap(it)

    def parse_named_colors(self, path: Path):
        text = Path(path).read_text()
        colors = {}
        # simple regex for lines like: name = hsv360 { 0 0 92 } or name = rgb { 1 0 1 }
        pattern = re.compile(
            r"^\s*([a-zA-Z0-9_]+)\s*=\s*(hsv360|rgb)\s*\{\s*([^}]+)\}",
            re.IGNORECASE | re.MULTILINE,
        )
        for m in pattern.finditer(text):
            name = m.group(1)
            typ = m.group(2).lower()
            vals = m.group(3).strip()
            parts = re.split(r"\s+", vals)
            try:
                if typ == "rgb":
                    # values 0-1
                    r = float(parts[0])
                    g = float(parts[1])
                    b = float(parts[2])
                    colors[name] = (r, g, b)
                elif typ == "hsv360":
                    h = float(parts[0])
                    s = float(parts[1])
                    v = float(parts[2])
                    # convert to 0-1 for colorsys: h/360, s/100, v/100
                    rf, gf, bf = colorsys.hsv_to_rgb(h / 360.0, s / 100.0, v / 100.0)
                    colors[name] = (rf, gf, bf)
            except Exception:
                continue
        return colors

    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, e):
        e.accept()

    # override view drop behaviour by installing event filter
    def eventFilter(self, obj, event):
        return False

    def apply_color_to_selected(self):
        # deprecated: removed Apply button; emblem pickers update immediately
        return

    def scale_changed(self, val):
        # kept for backward compatibility: set both axes
        sel = [i for i in self.scene.selectedItems() if isinstance(i, EmblemItem)]
        if not sel:
            return
        item = sel[0]
        s = val / 100.0
        item.set_scale_factor(s)
        self.update_emblem_pixmap(item)

    def scale_x_changed(self, val):
        sel = [i for i in self.scene.selectedItems() if isinstance(i, EmblemItem)]
        if not sel:
            return
        item = sel[0]
        sx = val / 100.0
        item.set_scale_x(sx)
        self.update_emblem_pixmap(item)

    def scale_y_changed(self, val):
        sel = [i for i in self.scene.selectedItems() if isinstance(i, EmblemItem)]
        if not sel:
            return
        item = sel[0]
        sy = val / 100.0
        item.set_scale_y(sy)
        self.update_emblem_pixmap(item)

    def mirror_x_toggled(self, checked):
        sel = [i for i in self.scene.selectedItems() if isinstance(i, EmblemItem)]
        if not sel:
            return
        for item in sel:
            item.mirror_x = bool(checked)
            self.update_emblem_pixmap(item)

    def mirror_y_toggled(self, checked):
        sel = [i for i in self.scene.selectedItems() if isinstance(i, EmblemItem)]
        if not sel:
            return
        for item in sel:
            item.mirror_y = bool(checked)
            self.update_emblem_pixmap(item)

    def export_coa(self):
        # collect current pattern filename (if any)
        # use selected pattern if present, otherwise prompt
        pattern_name = None
        if getattr(self, "current_pattern_name", None):
            pattern_name = self.current_pattern_name
            # default CoA key (tag) — user indicated tag is not important
            name = 'TAG'
        # build block
        lines = [f"{name} = {{"]
        # pattern: prompt user for file name or leave blank
        if pattern_name:
            lines.append(f'    pattern = "{pattern_name}"')
        else:
            pat, ok3 = QInputDialog.getText(
                self,
                "Pattern filename",
                "Enter pattern filename (include extension), or leave blank",
            )
            if pat:
                lines.append(f'    pattern = "{pat}"')
        # colors: export picked colors as rgb { r g b }
        qc1 = getattr(self.col1_btn, "_qcolor", None)
        qc2 = getattr(self.col2_btn, "_qcolor", None)
        qc3 = getattr(self.col3_btn, "_qcolor", None)

        def qc_to_rgb_tuple(qc):
            if not qc:
                return None
            return (qc.red(), qc.green(), qc.blue())

        pattern_c1 = qc_to_rgb_tuple(qc1)
        pattern_c2 = qc_to_rgb_tuple(qc2)
        pattern_c3 = qc_to_rgb_tuple(qc3)

        def fmt_rgb_line(name, rgb):
            if rgb is None:
                return None
            r = int(rgb[0])
            g = int(rgb[1])
            b = int(rgb[2])
            return f"    {name} = rgb {{ {r} {g} {b} }}"

        # determine how many pattern slots are used (1..3)
        try:
            if getattr(self, "_current_pattern_pil", None):
                # use robust detection which checks color proximity and frequencies
                pattern_slots = self._determine_pattern_slots(
                    self._current_pattern_pil, getattr(self, "_current_pattern_masks", None)
                )
            else:
                pattern_slots = 1
        except Exception:
            pattern_slots = 1

        # emit pattern color lines only for slots actually used
        if pattern_slots >= 1:
            c1_line = fmt_rgb_line("color1", pattern_c1)
            if c1_line:
                lines.append(c1_line)
        if pattern_slots >= 2:
            c2_line = fmt_rgb_line("color2", pattern_c2)
            if c2_line:
                lines.append(c2_line)
        if pattern_slots >= 3:
            c3_line = fmt_rgb_line("color3", pattern_c3)
            if c3_line:
                lines.append(c3_line)
        # iterate through scene items in z-order (lowest to highest), excluding background
        emblems = [i for i in self.scene.items() if isinstance(i, EmblemItem)]
        # sort by zValue to preserve layering
        emblems.sort(key=lambda e: e.zValue())
        for e in emblems:
            tex = e.image_path.name
            if getattr(e, "texture_type", "colored") == "colored":
                lines.append("    colored_emblem = {")
                lines.append(f'        texture = "{tex}"')
                # emblem colors: export explicit rgb by default
                ev = e.color_vals
                try:
                    inferred_emblem = infer_emblem_mask_colors(e.base_image)
                except Exception:
                    inferred_emblem = [None, None, None]

                saved = getattr(self, "emblem_saved_colors", {}).get(tex)
                # prefer explicit per-item colors to determine how many slots this emblem uses
                explicit_count = len([c for c in ev if c])
                if explicit_count > 0:
                    emblem_slots = explicit_count
                else:
                    # fall back to saved per-texture
                    saved_count = len([c for c in saved if c]) if saved else 0
                    if saved_count > 0:
                        emblem_slots = saved_count
                    else:
                        # fall back to inferred emblem mask colors
                        inferred_count = (
                            len([c for c in inferred_emblem if c])
                            if inferred_emblem
                            else 0
                        )
                        emblem_slots = inferred_count if inferred_count > 0 else 1

                # export emblem colours for this emblem instance according to emblem_slots
                for idx in range(emblem_slots):
                    pname = ("color1", "color2", "color3")[idx]
                    # priority: explicit item color -> saved per-texture -> inferred emblem mask
                    rgb_to_use = None
                    if idx < len(ev) and ev[idx]:
                        rgb_to_use = ev[idx]
                    elif saved and idx < len(saved) and saved[idx]:
                        rgb_to_use = saved[idx]
                    elif inferred_emblem and idx < len(inferred_emblem) and inferred_emblem[idx]:
                        rgb_to_use = inferred_emblem[idx]

                    if not rgb_to_use:
                        continue
                    r = int(rgb_to_use[0])
                    g = int(rgb_to_use[1])
                    b = int(rgb_to_use[2])
                    lines.append(f"        {pname} = rgb {{ {r} {g} {b} }}")
            else:
                # textured emblem: add textured_emblem block without color overrides
                lines.append("    textured_emblem = {")
                lines.append(f'        texture = "{tex}"')
            # instance
            pos = e.pos()
            # position is relative to top-left; convert to normalized center
            w = e.pixmap().width()
            h = e.pixmap().height()
            center_x = (pos.x() + w / 2) / CANVAS_W
            center_y = (pos.y() + h / 2) / CANVAS_H
            # use per-axis scales when exporting
            scale_x = getattr(e, "scale_x", getattr(e, "scale_factor", 1.0))
            scale_y = getattr(e, "scale_y", getattr(e, "scale_factor", 1.0))
            lines.append("        instance = {")
            lines.append(f"            position = {{ {center_x:.3f} {center_y:.3f} }}")
            lines.append(f"            scale = {{ {scale_x:.3f} {scale_y:.3f} }}")
            lines.append("        }")
            lines.append("    }")
        lines.append("}")
        # copy the CoA block to the clipboard instead of writing to a file
        text = "\n".join(lines)
        try:
            QApplication.clipboard().setText(text)
            QMessageBox.information(self, "Copied", "CoA block copied to clipboard")
        except Exception:
            QMessageBox.information(
                self, "Exported", "Generated CoA block (clipboard copy failed)"
            )

    def on_selection_changed(self):
        try:
            sel = [i for i in self.scene.selectedItems() if isinstance(i, EmblemItem)]
        except RuntimeError:
            return
        if not sel:
            return
        item = sel[0]
        # update scale slider to match item
        # sync X/Y scale controls with selected item
        try:
            self.scale_x_spin.blockSignals(True)
            self.scale_y_spin.blockSignals(True)
            self.scale_x_slider.blockSignals(True)
            self.scale_y_slider.blockSignals(True)
            sx = int(getattr(item, "scale_x", getattr(item, "scale_factor", 1.0)) * 100)
            sy = int(getattr(item, "scale_y", getattr(item, "scale_factor", 1.0)) * 100)
            self.scale_x_spin.setValue(sx)
            self.scale_x_slider.setValue(sx)
            self.scale_y_spin.setValue(sy)
            self.scale_y_slider.setValue(sy)
        finally:
            try:
                self.scale_x_spin.blockSignals(False)
                self.scale_y_spin.blockSignals(False)
                self.scale_x_slider.blockSignals(False)
                self.scale_y_slider.blockSignals(False)
            except Exception:
                pass
        # update mirror button states for selected emblem
        try:
            if getattr(item, "texture_type", "colored") != "colored":
                # textured emblems can still be mirrored visually
                pass
            self.mirror_x_btn.setChecked(bool(getattr(item, "mirror_x", False)))
            self.mirror_y_btn.setChecked(bool(getattr(item, "mirror_y", False)))
        except Exception:
            pass
        # textured emblems have no colors: hide controls and disable swatches
        if getattr(item, "texture_type", "colored") != "colored":
            self.update_emblem_color_controls(0)
            for b in (self.em_col1_btn, self.em_col2_btn, self.em_col3_btn):
                b.setEnabled(False)
            return

        try:
            inferred_slots = infer_emblem_mask_colors(item.base_image)
            num_slots = len([s for s in inferred_slots if s])
            if num_slots == 0:
                num_slots = 1
        except Exception:
            num_slots = 1

        # show/hide emblem controls accordingly
        self.update_emblem_color_controls(num_slots)

        # enable and set swatches for visible emblem buttons
        global_c1 = getattr(self.col1_btn, "_qcolor", QColor(255, 255, 255))
        global_c2 = getattr(self.col2_btn, "_qcolor", QColor(255, 255, 255))
        inferred = infer_emblem_mask_colors(item.base_image) if item.base_image else MASK_COLORS
        for idx, b in enumerate((self.em_col1_btn, self.em_col2_btn, self.em_col3_btn)):
            if not b.isVisible():
                b.setEnabled(False)
                continue
            b.setEnabled(True)
            cv = item.color_vals[idx]
            if cv:
                qc = QColor(int(cv[0]), int(cv[1]), int(cv[2]))
            else:
                if inferred and idx < len(inferred) and inferred[idx]:
                    qc = QColor(int(inferred[idx][0]), int(inferred[idx][1]), int(inferred[idx][2]))
                else:
                    if idx == 0:
                        qc = global_c1
                    elif idx == 1:
                        qc = global_c2
                    else:
                        qc = QColor(255, 255, 255)
            b._qcolor = qc
            b.setStyleSheet(f"background-color: {qc.name()};")

    def center_selected(self, axis: str = "both"):
        """Center the selected emblem.

        axis: 'both', 'x', or 'y'
        """
        sel = [i for i in self.scene.selectedItems() if isinstance(i, EmblemItem)]
        if not sel:
            QMessageBox.information(self, "Center", "Select an emblem first")
            return
        item = sel[0]
        cur = item.pos()
        w = item.pixmap().width()
        h = item.pixmap().height()
        # compute top-left target based on axis
        if axis == "both":
            tx = CANVAS_W / 2 - w / 2
            ty = CANVAS_H / 2 - h / 2
        elif axis == "x":
            tx = CANVAS_W / 2 - w / 2
            ty = cur.y()
        elif axis == "y":
            tx = cur.x()
            ty = CANVAS_H / 2 - h / 2
        else:
            tx = cur.x()
            ty = cur.y()
        item.setPos(QPointF(tx, ty))

    def delete_selected(self):
        sel = [i for i in self.scene.selectedItems() if isinstance(i, EmblemItem)]
        if not sel:
            QMessageBox.information(self, "Delete", "Select an emblem first")
            return
        for item in sel:
            self.scene.removeItem(item)

    def render_emblem_pixmap(self, item: EmblemItem):
        """Render an emblem PIL -> QPixmap applying recolor and pattern masking if present."""
        # recolor emblem
        pil = item.base_image
        if item.texture_type == "colored" and any(
            c is not None for c in item.color_vals
        ):
            pil = recolor_colored_emblem(item.base_image, item.color_vals)
        # scale emblem according to per-axis scales
        sx = getattr(item, "scale_x", getattr(item, "scale_factor", 1.0))
        sy = getattr(item, "scale_y", getattr(item, "scale_factor", 1.0))
        if sx != 1.0 or sy != 1.0:
            neww = int(pil.width * sx)
            newh = int(pil.height * sy)
            if neww <= 0:
                neww = 1
            if newh <= 0:
                newh = 1
            pil = pil.resize((neww, newh), Image.Resampling.LANCZOS)

        # apply mirroring flags if set
        try:
            if getattr(item, "mirror_x", False):
                pil = ImageOps.mirror(pil)
            if getattr(item, "mirror_y", False):
                pil = ImageOps.flip(pil)
        except Exception:
            pass

        # NOTE: emblem masking by pattern disabled — emblems are drawn above the pattern

        return pil2pixmap(pil)

    def update_emblem_pixmap(self, item: EmblemItem):
        try:
            pix = self.render_emblem_pixmap(item)
            item.setPixmap(pix)
        except Exception:
            pass


def main():
    app = QApplication(sys.argv)
    win = MainWindow()

    # enable drops on view by customizing mime handling
    # implement drop on view via overriding methods
    def view_dragEnter(e):
        if e.mimeData().hasFormat("application/x-eu5-asset"):
            # suppressed debug
            e.acceptProposedAction()
        else:
            e.ignore()

    def view_dragMove(e):
        if e.mimeData().hasFormat("application/x-eu5-asset"):
            # debug
            # print('view_dragMove')
            e.acceptProposedAction()
        else:
            e.ignore()

    def view_drop(e):
        md = e.mimeData().data("application/x-eu5-asset")
        try:
            payload = json.loads(bytes(md).decode("utf-8"))
            path = Path(payload["path"])
            # suppressed debug
            typ = payload.get("type", "colored")
            pil = load_image(path)
            # create item
            it = EmblemItem(path, pil, texture_type=typ)
            # get event position in view coordinates (Qt6 has position())
            if hasattr(e, "position"):
                evpos = e.position()
                vx, vy = evpos.x(), evpos.y()
            else:
                vp = e.pos()
                vx, vy = vp.x(), vp.y()
            scene_pos = win.view.mapToScene(int(vx), int(vy))
            it.setPos(
                scene_pos - QPointF(it.pixmap().width() / 2, it.pixmap().height() / 2)
            )
            win.scene.addItem(it)
            e.acceptProposedAction()
        except Exception:
            e.ignore()

    win.show()
    sys.exit(app.exec())


# allow Delete key to remove selected emblem when main window has focus
def _install_delete_shortcut(win: MainWindow):
    # we can override keyPressEvent on the view
    orig_key = win.view.keyPressEvent

    def keyPressEvent(ev):
        if ev.key() == Qt.Key_Delete:
            win.delete_selected()
            return
        orig_key(ev)

    win.view.keyPressEvent = keyPressEvent


if __name__ == "__main__":
    main()
