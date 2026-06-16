"""
Neural Network Playground
--------------------------
A visual tool for building, training, and inspecting small neural networks.
Uses your custom nn backend (neuron, layer, Network, BackPropagation, losses).

Layout:
  Left   — network graph (click neurons to inspect/edit)
  Middle — decision boundary + loss graph
  Right  — controls (dataset, activation, lr, layers)
"""

import sys
import math
import random
import time

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QThread, QPoint, QPointF, pyqtSignal
from PyQt6.QtGui import (
    QColor, QFont, QPainter, QPainterPath, QPen, QBrush, QRadialGradient,
)
from PyQt6.QtWidgets import (
    QApplication, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPushButton, QScrollArea, QSizePolicy, QSlider,
    QVBoxLayout, QWidget,
)

from neuron import NEURAL
from layer import LAYERS
from Network import Network
from losses import MSE
from BackPropagation import Trainer


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def make_xor(n=200):
    """Two classes separated by XOR logic (classic non-linear problem)."""
    pts = []
    for _ in range(n):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        pts.append(([x, y], [1 if (x > 0) != (y > 0) else 0]))
    return pts


def make_circle(n=200):
    """Inner circle = class 1, outer ring = class 0."""
    pts = []
    for _ in range(n):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        pts.append(([x, y], [1 if math.sqrt(x**2 + y**2) < 0.6 else 0]))
    return pts


def make_spiral(n=100):
    """Two interleaved spirals — hardest of the three datasets."""
    pts = []
    for i in range(n):
        t = i / n
        angle = t * 3 * math.pi
        r = t * 0.9
        x1 =  r * math.cos(angle) + random.gauss(0, 0.05)
        y1 =  r * math.sin(angle) + random.gauss(0, 0.05)
        x2 = -r * math.cos(angle) + random.gauss(0, 0.05)
        y2 = -r * math.sin(angle) + random.gauss(0, 0.05)
        pts += [([x1, y1], [1]), ([x2, y2], [0])]
    return pts


DATASETS = {"XOR": make_xor, "Circle": make_circle, "Spiral": make_spiral}


def build_network(layer_sizes: list[int], activation: str) -> Network:
    """Build a fresh Network from a list of layer sizes and an activation name."""
    layers = []
    for i, size in enumerate(layer_sizes):
        n_in = 2 if i == 0 else layer_sizes[i - 1]
        # Output layer always uses sigmoid so predictions stay in [0, 1]
        act = "sigmoid" if i == len(layer_sizes) - 1 else activation
        neurons = [NEURAL(n_in, activation=act) for _ in range(size)]
        layers.append(LAYERS(neurons, n_in))
    return Network(layers)


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

BG       = QColor(13, 15, 23)
PANEL    = QColor(20, 23, 35)
ACCENT   = QColor(99, 179, 237)    # blue
PURPLE   = QColor(154, 117, 234)   # purple
GREEN    = QColor(72, 199, 142)    # positive / class 1
RED      = QColor(252, 110, 110)   # negative / class 0
NEURON   = QColor(45, 55, 82)
DIM      = QColor(70, 82, 105)
WHITE    = QColor(255, 255, 255)
GOLD     = QColor(255, 214, 70)


# ---------------------------------------------------------------------------
# Global stylesheet
# ---------------------------------------------------------------------------

STYLE = """
QWidget {
    background: #0d0f17;
    color: #c4cfdf;
    font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
    font-size: 13px;
}
QLabel { background: transparent; }

/* ── buttons ── */
QPushButton {
    border-radius: 7px;
    padding: 6px 12px;
    font-weight: 600;
    color: #c4cfdf;
    background: #1c2030;
    border: 1.5px solid #2a3048;
}
QPushButton:hover {
    background: #232840;
    border-color: #63b3ed;
    color: #ffffff;
}
QPushButton:checked {
    background: #0d2340;
    border: 1.5px solid #63b3ed;
    color: #63b3ed;
}
QPushButton[role="act"]:checked {
    background: #1e1040;
    border: 1.5px solid #9a75ea;
    color: #9a75ea;
}
QPushButton[role="train"]  { background: #1a3d2b; border-color: #38a169; color: #68d391; }
QPushButton[role="train"]:hover  { background: #22523a; }
QPushButton[role="pause"]  { background: #3d300a; border-color: #d69e2e; color: #f6c050; }
QPushButton[role="pause"]:hover  { background: #4a3a10; }
QPushButton[role="reset"]  { background: #3d1010; border-color: #e53e3e; color: #fc8080; }
QPushButton[role="reset"]:hover  { background: #4a1818; }
QPushButton[role="add"]    { background: #102030; border-color: #2b6cb0; color: #63b3ed; }
QPushButton[role="remove"] { background: #200d0d; border-color: #742a2a; color: #fc8080; }

/* ── sliders ── */
QSlider::groove:horizontal {
    height: 5px; background: #1e2235; border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 15px; height: 15px; margin: -5px 0;
    background: #ffffff; border-radius: 8px;
}
QSlider::sub-page:horizontal { background: #63b3ed; border-radius: 3px; }

/* ── scroll ── */
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #1a1e2e; width: 6px; border-radius: 3px; }
QScrollBar::handle:vertical { background: #3a4060; border-radius: 3px; }

/* ── line edits ── */
QLineEdit {
    background: #0d0f17;
    border: 1px solid #2a3048;
    border-radius: 5px;
    padding: 3px 7px;
    color: #c4cfdf;
}
QLineEdit:focus { border-color: #63b3ed; }

/* ── frames ── */
QFrame[frameShape="4"], QFrame[frameShape="5"] { color: #1e2235; }
"""


# ---------------------------------------------------------------------------
# Network canvas
# ---------------------------------------------------------------------------

class NetworkCanvas(QWidget):
    """
    Draws the network as a graph: layers as columns, neurons as circles,
    weights as colored lines.

    Signals:
        layer_changed(layer_idx, delta)  — user clicked +/- on a layer
        neuron_clicked(layer_idx, neuron_idx) — user clicked a neuron circle
    """

    layer_changed  = pyqtSignal(int, int)
    neuron_clicked = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.network      = None
        self.layer_sizes  = []
        self._positions   = []        # [layer_idx][neuron_idx] = (x, y)
        self._hover_neuron = (-1, -1)
        self._sel_neuron   = (-1, -1)
        self._hover_layer  = -1

        self.setMinimumWidth(340)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def set_network(self, net: Network, sizes: list[int]):
        self.network     = net
        self.layer_sizes = sizes
        self.update()

    def _build_positions(self) -> list:
        """Compute pixel (x, y) for every neuron based on widget size."""
        W, H = self.width(), self.height()
        n = len(self.layer_sizes)
        margin_x = 55
        gap_x = (W - 2 * margin_x) / max(n - 1, 1)
        positions = []
        for li, size in enumerate(self.layer_sizes):
            cx = int(margin_x + li * gap_x) if n > 1 else W // 2
            col = []
            for ni in range(size):
                t = (ni + 1) / (size + 1)
                cy = int(65 + t * (H - 95))
                col.append((cx, cy))
            positions.append(col)
        self._positions = positions
        return positions

    # ── painting ──────────────────────────────────────────────────────────

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, PANEL)

        if not self.layer_sizes or self.network is None:
            return

        pos = self._build_positions()
        self._draw_connections(p, pos)
        self._draw_neurons(p, pos, H)

    def _draw_connections(self, p: QPainter, pos: list):
        """Draw weight lines between adjacent layers."""
        for li in range(len(pos) - 1):
            for ai, (ax, ay) in enumerate(pos[li]):
                for bi, (bx, by) in enumerate(pos[li + 1]):
                    try:
                        w = self.network.layers[li + 1].main_nodes[bi].weight[ai]
                        w = max(-1.0, min(1.0, w))
                    except Exception:
                        w = 0.0

                    alpha = int(30 + abs(w) * 160)
                    color = QColor(72, 199, 142, alpha) if w >= 0 else QColor(252, 110, 110, alpha)
                    p.setPen(QPen(color, max(1, int(abs(w) * 2.5))))
                    p.drawLine(ax, ay, bx, by)

    def _draw_neurons(self, p: QPainter, pos: list, H: int):
        """Draw each neuron as a glowing circle, with selection/hover rings."""
        n_layers = len(pos)

        for li, col in enumerate(pos):
            # layer label at top
            label = "Input" if li == 0 else ("Output" if li == n_layers - 1 else f"Hidden {li}")
            p.setFont(QFont("Segoe UI", 8))
            p.setPen(DIM)
            if col:
                cx = col[0][0]
                p.drawText(cx - 32, 14, 64, 16, Qt.AlignmentFlag.AlignHCenter, label)
                p.drawText(cx - 20, 30, 40, 16, Qt.AlignmentFlag.AlignHCenter, str(len(col)))

            for ni, (cx, cy) in enumerate(col):
                # read neuron output for brightness
                val = 0.5
                try:
                    v = self.network.layers[li].main_nodes[ni].output
                    if v is not None:
                        val = float(max(0.0, min(1.0, v)))
                except Exception:
                    pass

                is_selected = self._sel_neuron == (li, ni)
                is_hovered  = self._hover_neuron == (li, ni)

                # outer ring
                if is_selected:
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.setPen(QPen(GOLD, 2.5))
                    p.drawEllipse(cx - 19, cy - 19, 38, 38)
                elif is_hovered:
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.setPen(QPen(QColor(160, 210, 255, 100), 1.5))
                    p.drawEllipse(cx - 18, cy - 18, 36, 36)

                # fill with radial gradient based on activation value
                r = int(NEURON.red()   + val * (ACCENT.red()   - NEURON.red()))
                g = int(NEURON.green() + val * (ACCENT.green() - NEURON.green()))
                b = int(NEURON.blue()  + val * (ACCENT.blue()  - NEURON.blue()))
                grad = QRadialGradient(cx, cy, 16)
                grad.setColorAt(0, QColor(r, g, b, 220))
                grad.setColorAt(1, QColor(r, g, b, 50))

                p.setBrush(QBrush(grad))
                p.setPen(QPen(GOLD if is_selected else ACCENT, 1.8 if is_selected else 1.2))
                p.drawEllipse(cx - 13, cy - 13, 26, 26)

            # show +/- buttons when hovering this layer column
            if li == self._hover_layer and col:
                cx = col[0][0]
                self._draw_layer_btn(p, cx, 46, "+", GREEN)
                self._draw_layer_btn(p, cx, H - 50, "−", RED)

    def _draw_layer_btn(self, p: QPainter, cx: int, y: int, symbol: str, color: QColor):
        p.setBrush(QBrush(color))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(cx - 13, y, 26, 20, 5, 5)
        p.setPen(WHITE)
        p.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        p.drawText(cx - 13, y, 26, 20, Qt.AlignmentFlag.AlignCenter, symbol)

    # ── mouse events ──────────────────────────────────────────────────────

    def mouseMoveEvent(self, e):
        mx, my = e.position().x(), e.position().y()
        prev_layer  = self._hover_layer
        prev_neuron = self._hover_neuron
        self._hover_layer  = -1
        self._hover_neuron = (-1, -1)

        for li, col in enumerate(self._positions):
            if not col:
                continue
            # layer column hover (for +/- buttons)
            if abs(mx - col[0][0]) < 30:
                self._hover_layer = li
            # individual neuron hover
            for ni, (cx, cy) in enumerate(col):
                if math.sqrt((mx - cx) ** 2 + (my - cy) ** 2) < 15:
                    self._hover_neuron = (li, ni)

        cursor = (Qt.CursorShape.PointingHandCursor
                  if self._hover_neuron != (-1, -1)
                  else Qt.CursorShape.ArrowCursor)
        self.setCursor(cursor)

        if self._hover_layer != prev_layer or self._hover_neuron != prev_neuron:
            self.update()

    def leaveEvent(self, _):
        self._hover_layer  = -1
        self._hover_neuron = (-1, -1)
        self.update()

    def mousePressEvent(self, e):
        mx, my = e.position().x(), e.position().y()
        H = self.height()

        # neuron click has priority
        for li, col in enumerate(self._positions):
            for ni, (cx, cy) in enumerate(col):
                if math.sqrt((mx - cx) ** 2 + (my - cy) ** 2) < 15:
                    if self._sel_neuron == (li, ni):
                        self._sel_neuron = (-1, -1)   # deselect
                    else:
                        self._sel_neuron = (li, ni)
                        self.neuron_clicked.emit(li, ni)
                    self.update()
                    return

        # layer +/- buttons
        for li, col in enumerate(self._positions):
            if not col:
                continue
            cx = col[0][0]
            if abs(mx - cx) < 13:
                if 46 <= my <= 66:
                    self.layer_changed.emit(li, 1)
                    return
                if H - 50 <= my <= H - 30:
                    self.layer_changed.emit(li, -1)
                    return


# ---------------------------------------------------------------------------
# Neuron inspector panel
# ---------------------------------------------------------------------------

class NeuronPanel(QWidget):
    """
    Floating panel that appears when you click a neuron.
    Shows the neuron's live output, bias, and all weights — all editable.
    """

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("""
            NeuronPanel {
                background: #13172a;
                border: 1px solid #2a3258;
                border-radius: 10px;
            }
            QLabel  { background: transparent; color: #c4cfdf; }
            QLineEdit {
                background: #0d0f17; border: 1px solid #2a3048;
                border-radius: 5px; padding: 3px 7px; color: #c4cfdf;
            }
            QLineEdit:focus { border-color: #63b3ed; }
            QPushButton {
                border-radius: 6px; padding: 4px 10px;
                font-weight: 600; color: white;
            }
        """)
        self.setFixedWidth(250)
        self.hide()

        self._layer_idx  = -1
        self._neuron_idx = -1
        self._network    = None
        self._w_edits    = []

        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 14)
        root.setSpacing(10)

        # header row
        hdr = QHBoxLayout()
        self.title = QLabel("Neuron")
        self.title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.title.setStyleSheet("color: #63b3ed;")
        close = QPushButton("✕")
        close.setFixedSize(22, 22)
        close.setStyleSheet("background: #1e2235; border: none; border-radius: 11px; padding: 0; color: #c4cfdf;")
        close.clicked.connect(self.hide)
        hdr.addWidget(self.title)
        hdr.addStretch()
        hdr.addWidget(close)
        root.addLayout(hdr)

        # live output + activation
        info_row = QHBoxLayout()
        info_row.addWidget(QLabel("Output:"))
        self.output_lbl = QLabel("—")
        self.output_lbl.setStyleSheet("color: #9a75ea; font-weight: bold;")
        self.act_lbl = QLabel("")
        self.act_lbl.setStyleSheet("color: #505a72; font-size: 11px;")
        info_row.addWidget(self.output_lbl)
        info_row.addStretch()
        info_row.addWidget(self.act_lbl)
        root.addLayout(info_row)

        root.addWidget(self._sep())

        # bias field
        bias_row = QHBoxLayout()
        bias_row.addWidget(QLabel("Bias:"))
        self.bias_edit = QLineEdit()
        self.bias_edit.setFixedWidth(100)
        bias_row.addStretch()
        bias_row.addWidget(self.bias_edit)
        root.addLayout(bias_row)

        root.addWidget(self._sep())

        # weights list (scrollable)
        w_title = QLabel("Weights")
        w_title.setStyleSheet("color: #505a72; font-size: 11px; font-weight: bold;")
        root.addWidget(w_title)

        self._w_container = QWidget()
        self._w_container.setStyleSheet("background: transparent;")
        self._w_lay = QVBoxLayout(self._w_container)
        self._w_lay.setSpacing(4)
        self._w_lay.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidget(self._w_container)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(130)
        root.addWidget(scroll)

        # apply button
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setStyleSheet("background: #1a3050; border: 1px solid #2b6cb0;")
        self.apply_btn.clicked.connect(self._apply)
        root.addWidget(self.apply_btn)

    def _sep(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setStyleSheet("color: #1e2235;")
        return f

    def show_neuron(self, network: Network, li: int, ni: int, pos: QPoint):
        """Populate the panel with data from neuron (li, ni) and position it."""
        self._network    = network
        self._layer_idx  = li
        self._neuron_idx = ni

        neuron = network.layers[li].main_nodes[ni]
        n_layers = len(network.layers)
        layer_name = "Input" if li == 0 else ("Output" if li == n_layers - 1 else f"Hidden {li}")

        self.title.setText(f"{layer_name}  ·  #{ni + 1}")
        self.act_lbl.setText(neuron.activation or "linear")
        self.bias_edit.setText(f"{neuron.bias:.6f}")

        out = neuron.output
        self.output_lbl.setText(f"{out:.4f}" if out is not None else "—")

        # rebuild weight rows
        while self._w_lay.count():
            w = self._w_lay.takeAt(0).widget()
            if w:
                w.deleteLater()
        self._w_edits = []

        for i, wval in enumerate(neuron.weight):
            row_w = QWidget()
            row_w.setStyleSheet("background: transparent;")
            row = QHBoxLayout(row_w)
            row.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(f"w{i}")
            lbl.setFixedWidth(24)
            lbl.setStyleSheet("color: #505a72; font-size: 11px;")
            edit = QLineEdit(f"{wval:.6f}")
            self._w_edits.append(edit)
            row.addWidget(lbl)
            row.addWidget(edit)
            self._w_lay.addWidget(row_w)

        # position panel to the right of the neuron, centred vertically
        self.adjustSize()
        x = pos.x() + 22
        y = pos.y() - self.height() // 2

        if parent := self.parent():
            x = min(x, parent.width()  - self.width()  - 8)
            y = max(8, min(y, parent.height() - self.height() - 8))

        self.move(x, y)
        self.show()
        self.raise_()

    def refresh_output(self, network: Network):
        """Called every tick to keep the output value fresh while training."""
        if not self.isVisible() or self._layer_idx < 0:
            return
        try:
            out = network.layers[self._layer_idx].main_nodes[self._neuron_idx].output
            self.output_lbl.setText(f"{out:.4f}" if out is not None else "—")
        except Exception:
            pass

    def _apply(self):
        """Push the edited weights and bias back to the neuron."""
        if self._network is None or self._layer_idx < 0:
            return
        neuron = self._network.layers[self._layer_idx].main_nodes[self._neuron_idx]
        try:
            neuron.weight = [float(e.text()) for e in self._w_edits]
            neuron.bias   = float(self.bias_edit.text())
            self.apply_btn.setStyleSheet("background: #1a4030; border: 1px solid #38a169;")
        except ValueError:
            self.apply_btn.setStyleSheet("background: #301010; border: 1px solid #e53e3e;")
        QTimer.singleShot(700, lambda: self.apply_btn.setStyleSheet(
            "background: #1a3050; border: 1px solid #2b6cb0;"))


# ---------------------------------------------------------------------------
# Decision boundary canvas
# ---------------------------------------------------------------------------

GRID = 48   # resolution of the boundary grid (higher = slower)

class BoundaryCanvas(QWidget):
    """
    Draws a colour-coded grid showing the network's class prediction at every
    point in the 2D input space, plus the dataset points on top.
    """

    def __init__(self):
        super().__init__()
        self.network = None
        self.dataset = []
        self._grid   = None    # numpy array (GRID, GRID, 3) of RGB values
        self.setMinimumSize(360, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(self, net: Network, dataset: list):
        self.network = net
        self.dataset = dataset

    def recompute(self):
        """Re-run inference over the whole grid. Called from the main thread."""
        if self.network is None:
            return
        grid = np.zeros((GRID, GRID, 3), dtype=np.uint8)
        for gy in range(GRID):
            for gx in range(GRID):
                wx =  -1 + (gx + 0.5) * (2 / GRID)
                wy =   1 - (gy + 0.5) * (2 / GRID)
                try:
                    v = float(max(0.0, min(1.0, self.network.predict([wx, wy])[0])))
                except Exception:
                    v = 0.5
                # blend red → green
                grid[gy, gx] = [
                    int(252 * (1 - v) + 72 * v),
                    int(110 * (1 - v) + 199 * v),
                    int(110 * (1 - v) + 142 * v),
                ]
        self._grid = grid
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        pad  = 12
        size = min(W, H) - 2 * pad
        ox   = (W - size) // 2 + pad
        oy   = (H - size) // 2 + pad

        p.fillRect(0, 0, W, H, BG)

        # grid cells
        if self._grid is not None:
            cell = size / GRID
            for gy in range(GRID):
                for gx in range(GRID):
                    r, g, b = self._grid[gy, gx]
                    p.fillRect(
                        int(ox + gx * cell), int(oy + gy * cell),
                        math.ceil(cell) + 1, math.ceil(cell) + 1,
                        QColor(int(r), int(g), int(b)),
                    )

        # border
        p.setPen(QPen(DIM, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(ox, oy, size, size)

        # dataset points
        for (xy, lbl) in self.dataset:
            px = int(ox + (xy[0] + 1) / 2 * size)
            py = int(oy + (1 - (xy[1] + 1) / 2) * size)
            c  = GREEN if lbl[0] == 1 else RED
            p.setBrush(QBrush(c))
            p.setPen(QPen(QColor(255, 255, 255, 160), 1))
            p.drawEllipse(px - 5, py - 5, 10, 10)


# ---------------------------------------------------------------------------
# Loss graph
# ---------------------------------------------------------------------------

class LossGraph(QWidget):
    """Small line chart showing training loss over the last N epochs."""

    MAX_POINTS = 300

    def __init__(self):
        super().__init__()
        self.history: list[float] = []
        self.setFixedHeight(88)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def push(self, loss: float):
        self.history.append(loss)
        if len(self.history) > self.MAX_POINTS:
            self.history.pop(0)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, PANEL)

        if len(self.history) < 2:
            p.setPen(DIM)
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(0, 0, W, H, Qt.AlignmentFlag.AlignCenter, "Loss graph will appear here")
            return

        mx = max(self.history) or 1.0
        pts = [
            QPointF(
                i / (len(self.history) - 1) * W,
                H - 8 - (l / mx) * (H - 16),
            )
            for i, l in enumerate(self.history)
        ]

        # filled area under the curve
        fill = QPainterPath()
        fill.moveTo(pts[0])
        for pt in pts[1:]:
            fill.lineTo(pt)
        fill.lineTo(W, H)
        fill.lineTo(0, H)
        fill.closeSubpath()
        p.setBrush(QBrush(QColor(154, 117, 234, 28)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPath(fill)

        # line
        line = QPainterPath(pts[0])
        for pt in pts[1:]:
            line.lineTo(pt)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(PURPLE, 1.8))
        p.drawPath(line)

        # label
        p.setPen(DIM)
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(6, 14, f"loss  {self.history[-1]:.4f}")


# ---------------------------------------------------------------------------
# Training thread
# ---------------------------------------------------------------------------

class TrainWorker(QThread):
    """
    Runs training in a background thread so the UI stays responsive.
    Emits step_done(loss, epoch) after each batch of steps.
    """

    step_done = pyqtSignal(float, int)

    def __init__(self, trainer: Trainer, dataset: list, speed_fn):
        super().__init__()
        self.trainer   = trainer
        self.dataset   = dataset
        self.speed_fn  = speed_fn   # callable → int (steps per tick)
        self._running  = True
        self.epoch     = 0

    def stop(self):
        self._running = False

    def run(self):
        Xs = [d[0] for d in self.dataset]
        Ys = [d[1] for d in self.dataset]
        while self._running:
            steps = max(1, int(self.speed_fn()))
            loss  = 0.0
            for _ in range(steps):
                loss = self.trainer.train_epoch(Xs, Ys)
            self.epoch += steps
            self.step_done.emit(loss, self.epoch)
            time.sleep(0.01)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """
    Root window — assembles all panels and wires them together.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Network Playground")
        self.resize(1300, 820)
        self.setStyleSheet(STYLE)

        # state
        self.layer_sizes  = [4, 4, 1]
        self.activation   = "sigmoid"
        self.dataset_name = "XOR"
        self.dataset      = make_xor()
        self.network      = build_network(self.layer_sizes, self.activation)
        self.loss_fn      = MSE()
        self.trainer      = None
        self.worker       = None

        self._build_ui()
        self._rebuild_network()

        self._tick_timer = QTimer()
        self._tick_timer.timeout.connect(self._tick)
        self._tick_timer.start(100)

    # ── layout ────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setSpacing(0)
        main.setContentsMargins(0, 0, 0, 0)

        # left: network graph
        self.net_canvas = NetworkCanvas()
        self.net_canvas.layer_changed.connect(self._on_layer_changed)
        self.net_canvas.neuron_clicked.connect(self._on_neuron_clicked)
        main.addWidget(self.net_canvas, 3)

        # neuron inspector (floats over the left panel)
        self.neuron_panel = NeuronPanel(self)

        main.addWidget(self._vline())

        # middle: boundary + loss
        mid = QWidget()
        mid_lay = QVBoxLayout(mid)
        mid_lay.setContentsMargins(0, 0, 0, 0)
        mid_lay.setSpacing(0)

        self.status_lbl = QLabel("Ready")
        self.status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_lbl.setStyleSheet("color: #63b3ed; padding: 6px; font-size: 12px;")
        mid_lay.addWidget(self.status_lbl)

        self.boundary = BoundaryCanvas()
        mid_lay.addWidget(self.boundary, 1)

        self.loss_graph = LossGraph()
        mid_lay.addWidget(self.loss_graph)

        main.addWidget(mid, 4)
        main.addWidget(self._vline())

        # right: controls
        main.addWidget(self._build_controls())

    def _build_controls(self) -> QWidget:
        """Build the right-hand controls panel."""
        panel = QWidget()
        panel.setFixedWidth(290)
        panel.setStyleSheet("background: #111422;")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(14)

        # title
        t = QLabel("Controls")
        t.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        t.setStyleSheet("color: #c4cfdf;")
        lay.addWidget(t)

        # train / pause / reset
        row = QHBoxLayout()
        self.btn_train = QPushButton("▶  Train")
        self.btn_pause = QPushButton("⏸  Pause")
        self.btn_reset = QPushButton("↺  Reset")
        self.btn_train.setProperty("role", "train")
        self.btn_pause.setProperty("role", "pause")
        self.btn_reset.setProperty("role", "reset")
        self.btn_train.clicked.connect(self._start_training)
        self.btn_pause.clicked.connect(self._stop_training)
        self.btn_reset.clicked.connect(self._rebuild_network)
        for b in (self.btn_train, self.btn_pause, self.btn_reset):
            row.addWidget(b)
        lay.addLayout(row)

        lay.addWidget(self._hline())

        # dataset
        lay.addWidget(self._section("Dataset"))
        ds_row = QHBoxLayout()
        ds_row.setSpacing(6)
        self._ds_btns: dict[str, QPushButton] = {}
        for name in DATASETS:
            b = QPushButton(name)
            b.setCheckable(True)
            b.setChecked(name == self.dataset_name)
            b.clicked.connect(lambda _, n=name: self._set_dataset(n))
            ds_row.addWidget(b)
            self._ds_btns[name] = b
        lay.addLayout(ds_row)

        lay.addWidget(self._hline())

        # activation
        lay.addWidget(self._section("Activation  (hidden layers)"))
        act_grid = QGridLayout()
        act_grid.setSpacing(6)
        self._act_btns: dict[str, QPushButton] = {}
        for i, name in enumerate(["sigmoid", "relu", "tanh", "leaky_relu", "linear"]):
            b = QPushButton(name)
            b.setCheckable(True)
            b.setChecked(name == self.activation)
            b.setProperty("role", "act")
            b.clicked.connect(lambda _, n=name: self._set_activation(n))
            act_grid.addWidget(b, i // 2, i % 2)
            self._act_btns[name] = b
        lay.addLayout(act_grid)

        lay.addWidget(self._hline())

        # learning rate
        lay.addWidget(self._section("Learning Rate"))
        lr_row = QHBoxLayout()
        self.lr_slider = QSlider(Qt.Orientation.Horizontal)
        self.lr_slider.setRange(1, 1000)
        self.lr_slider.setValue(30)
        self.lr_val = QLabel("0.030")
        self.lr_val.setFixedWidth(42)
        self.lr_val.setStyleSheet("color: #63b3ed;")
        self.lr_slider.valueChanged.connect(self._on_lr_change)
        lr_row.addWidget(self.lr_slider)
        lr_row.addWidget(self.lr_val)
        lay.addLayout(lr_row)

        # speed
        lay.addWidget(self._section("Steps / frame"))
        sp_row = QHBoxLayout()
        self.sp_slider = QSlider(Qt.Orientation.Horizontal)
        self.sp_slider.setRange(1, 50)
        self.sp_slider.setValue(10)
        self.sp_val = QLabel("10")
        self.sp_val.setFixedWidth(28)
        self.sp_val.setStyleSheet("color: #63b3ed;")
        self.sp_slider.valueChanged.connect(lambda v: self.sp_val.setText(str(v)))
        sp_row.addWidget(self.sp_slider)
        sp_row.addWidget(self.sp_val)
        lay.addLayout(sp_row)

        lay.addWidget(self._hline())

        # layer editor
        lay.addWidget(self._section("Layers  (+ / − neurons per layer)"))
        self.layer_editor = QWidget()
        self.layer_editor.setStyleSheet("background: transparent;")
        self._layer_editor_lay = QHBoxLayout(self.layer_editor)
        self._layer_editor_lay.setSpacing(6)
        lay.addWidget(self.layer_editor)
        self._refresh_layer_editor()

        # add / remove layer
        ar = QHBoxLayout()
        ar.setSpacing(8)
        b_add = QPushButton("+ Add Layer")
        b_rem = QPushButton("− Remove Layer")
        b_add.setProperty("role", "add")
        b_rem.setProperty("role", "remove")
        b_add.clicked.connect(self._add_layer)
        b_rem.clicked.connect(self._remove_layer)
        ar.addWidget(b_add)
        ar.addWidget(b_rem)
        lay.addLayout(ar)

        lay.addStretch()
        return panel

    # ── small helpers ─────────────────────────────────────────────────────

    def _vline(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.VLine)
        return f

    def _hline(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        return f

    def _section(self, text: str) -> QLabel:
        l = QLabel(text)
        l.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        l.setStyleSheet("color: #404a62; letter-spacing: 0.8px;")
        return l

    # ── layer editor ──────────────────────────────────────────────────────

    def _refresh_layer_editor(self):
        """Rebuild the per-layer neuron count boxes in the controls panel."""
        while self._layer_editor_lay.count():
            w = self._layer_editor_lay.takeAt(0).widget()
            if w:
                w.deleteLater()

        n = len(self.layer_sizes)
        for i, size in enumerate(self.layer_sizes):
            box = QWidget()
            box.setStyleSheet("background: #181d30; border-radius: 8px;")
            vl = QVBoxLayout(box)
            vl.setContentsMargins(4, 5, 4, 5)
            vl.setSpacing(2)

            plus = QPushButton("+")
            plus.setFixedSize(28, 22)
            plus.setStyleSheet("background: #1a4030; border: 1px solid #38a169; border-radius: 5px; padding: 0; color: #68d391; font-weight: bold;")
            plus.clicked.connect(lambda _, idx=i: self._on_layer_changed(idx, 1))

            num = QLabel(str(size))
            num.setAlignment(Qt.AlignmentFlag.AlignCenter)
            num.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
            num.setStyleSheet("color: #c4cfdf;")

            minus = QPushButton("−")
            minus.setFixedSize(28, 22)
            minus.setStyleSheet("background: #301010; border: 1px solid #742a2a; border-radius: 5px; padding: 0; color: #fc8080; font-weight: bold;")
            minus.clicked.connect(lambda _, idx=i: self._on_layer_changed(idx, -1))

            name_lbl = QLabel("In" if i == 0 else ("Out" if i == n - 1 else f"H{i}"))
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_lbl.setStyleSheet("color: #404a62; font-size: 9px;")

            vl.addWidget(plus,     alignment=Qt.AlignmentFlag.AlignHCenter)
            vl.addWidget(num,      alignment=Qt.AlignmentFlag.AlignHCenter)
            vl.addWidget(minus,    alignment=Qt.AlignmentFlag.AlignHCenter)
            vl.addWidget(name_lbl, alignment=Qt.AlignmentFlag.AlignHCenter)

            self._layer_editor_lay.addWidget(box)

    # ── slots ─────────────────────────────────────────────────────────────

    def _on_layer_changed(self, idx: int, delta: int):
        self.layer_sizes[idx] = max(1, min(8, self.layer_sizes[idx] + delta))
        self._rebuild_network()

    def _add_layer(self):
        if len(self.layer_sizes) < 7:
            self.layer_sizes.insert(-1, 4)
            self._rebuild_network()

    def _remove_layer(self):
        if len(self.layer_sizes) > 1:
            self.layer_sizes.pop(-2)
            self._rebuild_network()

    def _set_dataset(self, name: str):
        self.dataset_name = name
        self.dataset = DATASETS[name]()
        for n, b in self._ds_btns.items():
            b.setChecked(n == name)
            b.style().unpolish(b)
            b.style().polish(b)
        self._rebuild_network()

    def _set_activation(self, name: str):
        self.activation = name
        for n, b in self._act_btns.items():
            b.setChecked(n == name)
            b.style().unpolish(b)
            b.style().polish(b)
        self._rebuild_network()

    def _on_lr_change(self, v: int):
        lr = v / 1000
        self.lr_val.setText(f"{lr:.3f}")
        if self.trainer:
            self.trainer.lr = lr

    def _on_neuron_clicked(self, li: int, ni: int):
        """Show the neuron inspector panel next to the clicked neuron."""
        x, y = self.net_canvas._positions[li][ni]
        global_pt  = self.net_canvas.mapToGlobal(QPoint(int(x), int(y)))
        local_pt   = self.mapFromGlobal(global_pt)
        self.neuron_panel.show_neuron(self.network, li, ni, local_pt)

    def _rebuild_network(self):
        """Tear down and reconstruct the network, reset all state."""
        self._stop_training()
        if hasattr(self, "neuron_panel"):
            self.neuron_panel.hide()
        if hasattr(self, "net_canvas"):
            self.net_canvas._sel_neuron = (-1, -1)

        self.network  = build_network(self.layer_sizes, self.activation)
        self.loss_fn  = MSE()
        self.trainer  = None

        if hasattr(self, "loss_graph"):
            self.loss_graph.history.clear()
            self.loss_graph.update()
        if hasattr(self, "status_lbl"):
            self.status_lbl.setText("Ready")
        if hasattr(self, "boundary"):
            self.boundary.set_data(self.network, self.dataset)
            self.boundary.recompute()
        if hasattr(self, "net_canvas"):
            self.net_canvas.set_network(self.network, self.layer_sizes)
        if hasattr(self, "_layer_editor_lay"):
            self._refresh_layer_editor()

    def _start_training(self):
        if self.worker and self.worker.isRunning():
            return
        self.trainer = Trainer(self.network, self.loss_fn, self.lr_slider.value() / 1000)
        self.worker  = TrainWorker(self.trainer, self.dataset, lambda: self.sp_slider.value())
        self.worker.step_done.connect(self._on_step)
        self.worker.start()
        self.status_lbl.setText("Training…")

    def _stop_training(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
        if hasattr(self, "status_lbl"):
            self.status_lbl.setText("Paused")

    def _on_step(self, loss: float, epoch: int):
        self.loss_graph.push(loss)
        self.status_lbl.setText(f"Epoch {epoch}   loss {loss:.4f}")

    def _tick(self):
        """Called every 100 ms to refresh the boundary and neuron output."""
        if self.worker and self.worker.isRunning():
            self.boundary.recompute()
        self.net_canvas.update()
        if hasattr(self, "neuron_panel"):
            self.neuron_panel.refresh_output(self.network)

    def closeEvent(self, e):
        self._stop_training()
        e.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()