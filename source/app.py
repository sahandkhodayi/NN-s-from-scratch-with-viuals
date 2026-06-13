import sys, math, random, threading, time
sys.path.insert(0, '/home/claude')

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QSlider, QComboBox, QFrame, QSizePolicy,
    QButtonGroup, QGridLayout, QSpinBox
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QLinearGradient,
    QFont, QPainterPath, QRadialGradient
)
import numpy as np

from neuron import NEURAL
from layer import LAYERS
from Network import Network
from losses import MSE
from BackPropagation import Trainer

# ── Datasets ──────────────────────────────────────────────────────────────────
def make_xor(n=200):
    pts = []
    for _ in range(n):
        x, y = random.uniform(-1,1), random.uniform(-1,1)
        pts.append(([x, y], [1 if (x>0)!=(y>0) else 0]))
    return pts

def make_circle(n=200):
    pts = []
    for _ in range(n):
        x, y = random.uniform(-1,1), random.uniform(-1,1)
        pts.append(([x, y], [1 if math.sqrt(x**2+y**2)<0.6 else 0]))
    return pts

def make_spiral(n=100):
    pts = []
    for i in range(n):
        t = i/n; angle = t*3*math.pi; r = t*0.9
        x1 =  r*math.cos(angle)+random.gauss(0,.05)
        y1 =  r*math.sin(angle)+random.gauss(0,.05)
        x2 = -r*math.cos(angle)+random.gauss(0,.05)
        y2 = -r*math.sin(angle)+random.gauss(0,.05)
        pts += [([x1,y1],[1]), ([x2,y2],[0])]
    return pts

DATASETS = {"XOR": make_xor, "Circle": make_circle, "Spiral": make_spiral}

def build_network(layer_sizes, activation):
    layers = []
    for i, size in enumerate(layer_sizes):
        n_in = 2 if i==0 else layer_sizes[i-1]
        act  = "sigmoid" if i==len(layer_sizes)-1 else activation
        neurons = [NEURAL(n_in, activation=act) for _ in range(size)]
        layers.append(LAYERS(neurons, n_in))
    return Network(layers)

# ── Colors ────────────────────────────────────────────────────────────────────
C_BG       = QColor(15,17,26)
C_PANEL    = QColor(22,25,37)
C_ACCENT   = QColor(99,179,237)
C_ACCENT2  = QColor(154,117,234)
C_POS      = QColor(72,199,142)
C_NEG      = QColor(252,110,110)
C_TEXT     = QColor(200,210,230)
C_DIM      = QColor(80,90,110)
C_NEURON   = QColor(55,65,95)

STYLE = """
QWidget { background: #0f111a; color: #c8d2e6; font-family: 'Segoe UI', Arial; }
QLabel  { background: transparent; }
QPushButton {
    border-radius: 7px; padding: 6px 14px;
    font-weight: bold; font-size: 13px; color: white;
}
QPushButton:hover { opacity: 0.85; }
QSlider::groove:horizontal {
    height: 6px; background: #232838; border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 16px; height: 16px; margin: -5px 0;
    background: white; border-radius: 8px;
}
QSlider::sub-page:horizontal { background: #63b3ed; border-radius: 3px; }
QComboBox {
    background: #1a1e2e; border: 1px solid #3a4060;
    border-radius: 6px; padding: 4px 10px; color: #c8d2e6;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background: #1a1e2e; selection-background-color: #63b3ed; }
QFrame[frameShape="4"], QFrame[frameShape="5"] { color: #232838; }
"""

# ── Network Canvas ─────────────────────────────────────────────────────────────
class NetworkCanvas(QWidget):
    layer_changed = pyqtSignal(int, int)   # layer_idx, delta (+1/-1)
    layer_removed = pyqtSignal(int)
    layer_added   = pyqtSignal(int)        # insert before index

    def __init__(self):
        super().__init__()
        self.network     = None
        self.layer_sizes = [4, 4, 1]
        self.setMinimumWidth(360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._hover_layer = -1
        self._positions   = []   # [layer][neuron] = (x,y)

    def set_network(self, net, sizes):
        self.network     = net
        self.layer_sizes = sizes
        self.update()

    def _compute_positions(self):
        W, H     = self.width(), self.height()
        n        = len(self.layer_sizes)
        margin_x = 50
        spacing  = (W - 2*margin_x) / max(n-1, 1)
        pos      = []
        for li, size in enumerate(self.layer_sizes):
            cx   = int(margin_x + li * spacing) if n>1 else W//2
            lpos = []
            for ni in range(size):
                t  = (ni+1)/(size+1)
                cy = int(60 + t*(H-90))
                lpos.append((cx, cy))
            pos.append(lpos)
        self._positions = pos
        return pos

    def paintEvent(self, _):
        p   = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()

        # background
        p.fillRect(0, 0, W, H, C_PANEL)

        if not self.layer_sizes:
            return

        pos = self._compute_positions()
        net = self.network

        # ── connections ──
        for li in range(len(pos)-1):
            for ai, (ax,ay) in enumerate(pos[li]):
                for bi, (bx,by) in enumerate(pos[li+1]):
                    w_val = 0.0
                    try:
                        w_val = net.layers[li+1].main_nodes[bi].weight[ai]
                        w_val = max(-1, min(1, w_val))
                    except: pass
                    alpha = int(40 + abs(w_val)*180)
                    if w_val >= 0:
                        c = QColor(72,199,142, alpha)
                    else:
                        c = QColor(252,110,110, alpha)
                    thick = max(1, int(abs(w_val)*3))
                    p.setPen(QPen(c, thick))
                    p.drawLine(ax, ay, bx, by)

        # ── neurons ──
        for li, lpos in enumerate(pos):
            # label
            lbl = "Input" if li==0 else ("Output" if li==len(pos)-1 else f"Hidden {li}")
            p.setPen(C_DIM)
            p.setFont(QFont("Segoe UI", 9))
            if lpos:
                p.drawText(lpos[0][0]-30, 18, 60, 20, Qt.AlignmentFlag.AlignHCenter, lbl)
                p.drawText(lpos[0][0]-20, 32, 40, 18, Qt.AlignmentFlag.AlignHCenter, str(len(lpos)))

            for ni, (cx,cy) in enumerate(lpos):
                val = 0.5
                try:
                    v = net.layers[li].main_nodes[ni].output
                    if v is not None: val = max(0, min(1, float(v)))
                except: pass

                # glow
                grad = QRadialGradient(cx, cy, 18)
                r = int(C_NEURON.red()   + val*(C_ACCENT.red()  -C_NEURON.red()))
                g = int(C_NEURON.green() + val*(C_ACCENT.green()-C_NEURON.green()))
                b = int(C_NEURON.blue()  + val*(C_ACCENT.blue() -C_NEURON.blue()))
                grad.setColorAt(0, QColor(r,g,b,230))
                grad.setColorAt(1, QColor(r,g,b,60))
                p.setBrush(QBrush(grad))
                p.setPen(QPen(C_ACCENT, 1.5))
                p.drawEllipse(cx-13, cy-13, 26, 26)

            # hover: show +/- buttons
            if li == self._hover_layer:
                cx = lpos[0][0] if lpos else W//2
                # plus
                p.setBrush(QBrush(C_POS))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRoundedRect(cx-14, 48, 28, 22, 5, 5)
                p.setPen(QColor(255,255,255))
                p.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
                p.drawText(cx-14, 48, 28, 22, Qt.AlignmentFlag.AlignCenter, "+")
                # minus
                p.setBrush(QBrush(C_NEG))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRoundedRect(cx-14, H-52, 28, 22, 5, 5)
                p.setPen(QColor(255,255,255))
                p.drawText(cx-14, H-52, 28, 22, Qt.AlignmentFlag.AlignCenter, "−")

        # ── add-layer hint zones ──
        p.setFont(QFont("Segoe UI", 9))
        p.setPen(C_DIM)

    def mouseMoveEvent(self, e):
        pos   = self._positions
        mx    = e.position().x()
        hover = -1
        for li, lpos in enumerate(pos):
            if lpos:
                cx = lpos[0][0]
                if abs(mx - cx) < 30:
                    hover = li
                    break
        if hover != self._hover_layer:
            self._hover_layer = hover
            self.update()

    def leaveEvent(self, _):
        self._hover_layer = -1
        self.update()

    def mousePressEvent(self, e):
        pos = self._positions
        mx, my = e.position().x(), e.position().y()
        H = self.height()
        for li, lpos in enumerate(pos):
            if not lpos: continue
            cx = lpos[0][0]
            if abs(mx - cx) < 14:
                # plus button
                if 48 <= my <= 70:
                    self.layer_changed.emit(li, 1)
                    return
                # minus button
                if H-52 <= my <= H-30:
                    self.layer_changed.emit(li, -1)
                    return


# ── Decision Boundary Canvas ──────────────────────────────────────────────────
GRID = 50

class BoundaryCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.network  = None
        self.dataset  = []
        self._grid    = None   # numpy H×W×3
        self.setMinimumSize(380, 380)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(self, net, dataset):
        self.network = net
        self.dataset = dataset

    def recompute(self):
        if self.network is None:
            return
        net  = self.network
        grid = np.zeros((GRID, GRID, 3), dtype=np.uint8)
        for gy in range(GRID):
            for gx in range(GRID):
                wx =  -1 + (gx+.5)*(2/GRID)
                wy =   1 - (gy+.5)*(2/GRID)
                try:    v = max(0, min(1, net.predict([wx,wy])[0]))
                except: v = .5
                r = int(252*(1-v) + 72*v)
                g = int(110*(1-v) + 199*v)
                b = int(110*(1-v) + 142*v)
                grid[gy,gx] = [r,g,b]
        self._grid = grid
        self.update()

    def paintEvent(self, _):
        p    = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        pad  = 10
        size = min(W, H) - 2*pad
        ox   = (W - size)//2 + pad
        oy   = (H - size)//2 + pad

        p.fillRect(0, 0, W, H, C_BG)

        # grid
        if self._grid is not None:
            cell = size / GRID
            for gy in range(GRID):
                for gx in range(GRID):
                    r,g,b = self._grid[gy,gx]
                    p.fillRect(
                        int(ox + gx*cell), int(oy + gy*cell),
                        math.ceil(cell)+1, math.ceil(cell)+1,
                        QColor(int(r),int(g),int(b))
                    )

        # border
        p.setPen(QPen(C_DIM, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(ox, oy, size, size)

        # data points
        for (xy, lbl) in self.dataset:
            px = int(ox + (xy[0]+1)/2 * size)
            py = int(oy + (1-(xy[1]+1)/2) * size)
            c  = C_POS if lbl[0]==1 else C_NEG
            p.setBrush(QBrush(c))
            p.setPen(QPen(QColor(255,255,255,180), 1))
            p.drawEllipse(px-5, py-5, 10, 10)


# ── Loss Graph ────────────────────────────────────────────────────────────────
class LossGraph(QWidget):
    def __init__(self):
        super().__init__()
        self.history = []
        self.setFixedHeight(90)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def add(self, loss):
        self.history.append(loss)
        if len(self.history) > 300:
            self.history.pop(0)
        self.update()

    def paintEvent(self, _):
        p    = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, C_PANEL)

        if len(self.history) < 2:
            p.setPen(C_DIM)
            p.drawText(0,0,W,H, Qt.AlignmentFlag.AlignCenter, "Loss graph")
            return

        mx  = max(self.history) or 1
        pts = []
        for i, l in enumerate(self.history):
            x = int(i/(len(self.history)-1)*W)
            y = int(H - 8 - (l/mx)*(H-16))
            pts.append(QPointF(x, y))

        path = QPainterPath(pts[0])
        for pt in pts[1:]:
            path.lineTo(pt)

        pen = QPen(C_ACCENT2, 2)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

        # fill under
        fill = QPainterPath(path)
        fill.lineTo(W, H); fill.lineTo(0, H); fill.closeSubpath()
        p.setBrush(QBrush(QColor(154,117,234,30)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPath(fill)

        p.setPen(C_DIM)
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(4, 12, f"Loss: {self.history[-1]:.4f}")
        p.drawText(4, H-4, "Loss graph")


# ── Training Worker ───────────────────────────────────────────────────────────
class TrainWorker(QThread):
    step_done = pyqtSignal(float, int)   # loss, epoch

    def __init__(self, trainer, dataset, speed_fn):
        super().__init__()
        self.trainer  = trainer
        self.dataset  = dataset
        self.speed_fn = speed_fn
        self._running = True
        self.epoch    = 0

    def stop(self):
        self._running = False

    def run(self):
        Xs = [d[0] for d in self.dataset]
        Ys = [d[1] for d in self.dataset]
        while self._running:
            steps = max(1, int(self.speed_fn()))
            loss  = 0
            for _ in range(steps):
                loss = self.trainer.train_epoch(Xs, Ys)
            self.epoch += steps
            self.step_done.emit(loss, self.epoch)
            time.sleep(0.01)


# ── Main Window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Network Playground")
        self.resize(1280, 800)
        self.setStyleSheet(STYLE)

        self.layer_sizes  = [4, 4, 1]
        self.activation   = "sigmoid"
        self.dataset_name = "XOR"
        self.dataset      = make_xor()
        self.network      = build_network(self.layer_sizes, self.activation)
        self.loss_fn      = MSE()
        self.trainer      = None
        self.worker       = None

        self._build_ui()
        self._refresh_network()

        # timer to refresh canvases
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(100)

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        main = QHBoxLayout(root); main.setSpacing(0); main.setContentsMargins(0,0,0,0)

        # ── Left: network ──
        self.net_canvas = NetworkCanvas()
        self.net_canvas.layer_changed.connect(self._on_layer_changed)
        main.addWidget(self.net_canvas, 3)

        sep1 = QFrame(); sep1.setFrameShape(QFrame.Shape.VLine); main.addWidget(sep1)

        # ── Middle: boundary + loss ──
        mid = QWidget(); mid_lay = QVBoxLayout(mid)
        mid_lay.setContentsMargins(0,0,0,0); mid_lay.setSpacing(0)

        self.epoch_lbl = QLabel("Epoch: 0")
        self.epoch_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.epoch_lbl.setFont(QFont("Segoe UI", 10))
        self.epoch_lbl.setStyleSheet("color:#63b3ed; padding:6px;")
        mid_lay.addWidget(self.epoch_lbl)

        self.boundary = BoundaryCanvas()
        mid_lay.addWidget(self.boundary, 1)

        self.loss_graph = LossGraph()
        mid_lay.addWidget(self.loss_graph)

        main.addWidget(mid, 4)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.VLine); main.addWidget(sep2)

        # ── Right: controls ──
        ctrl = QWidget(); ctrl.setFixedWidth(280)
        ctrl.setStyleSheet("background:#161825;")
        ctrl_lay = QVBoxLayout(ctrl)
        ctrl_lay.setContentsMargins(16,16,16,16); ctrl_lay.setSpacing(14)

        title = QLabel("Controls")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color:#c8d2e6;")
        ctrl_lay.addWidget(title)

        # Train / Pause / Reset
        btn_row = QHBoxLayout(); btn_row.setSpacing(8)
        self.btn_train = QPushButton("▶  Train")
        self.btn_pause = QPushButton("⏸  Pause")
        self.btn_reset = QPushButton("↺  Reset")
        self.btn_train.setStyleSheet("background:#38a169;")
        self.btn_pause.setStyleSheet("background:#d69e2e;")
        self.btn_reset.setStyleSheet("background:#e53e3e;")
        self.btn_train.clicked.connect(self._start_training)
        self.btn_pause.clicked.connect(self._pause_training)
        self.btn_reset.clicked.connect(self._reset)
        for b in [self.btn_train, self.btn_pause, self.btn_reset]:
            btn_row.addWidget(b)
        ctrl_lay.addLayout(btn_row)

        self._sep(ctrl_lay)

        # Dataset
        ctrl_lay.addWidget(self._lbl("DATASET"))
        ds_row = QHBoxLayout(); ds_row.setSpacing(6)
        self._ds_btns = {}
        for name in DATASETS:
            b = QPushButton(name)
            b.setCheckable(True)
            b.setChecked(name == self.dataset_name)
            b.clicked.connect(lambda _, n=name: self._set_dataset(n))
            b.setStyleSheet("background:#28304a; border-radius:7px; padding:5px;")
            ds_row.addWidget(b)
            self._ds_btns[name] = b
        ctrl_lay.addLayout(ds_row)

        self._sep(ctrl_lay)

        # Activation
        ctrl_lay.addWidget(self._lbl("ACTIVATION (hidden layers)"))
        act_grid = QGridLayout(); act_grid.setSpacing(6)
        acts = ["sigmoid","relu","tanh","leaky_relu","linear"]
        self._act_btns = {}
        for i, name in enumerate(acts):
            b = QPushButton(name)
            b.setCheckable(True)
            b.setChecked(name == self.activation)
            b.clicked.connect(lambda _, n=name: self._set_activation(n))
            b.setStyleSheet("background:#28304a; border-radius:7px; padding:5px;")
            act_grid.addWidget(b, i//2, i%2)
            self._act_btns[name] = b
        ctrl_lay.addLayout(act_grid)

        self._sep(ctrl_lay)

        # Learning rate
        ctrl_lay.addWidget(self._lbl("Learning Rate"))
        lr_row = QHBoxLayout()
        self.lr_slider = QSlider(Qt.Orientation.Horizontal)
        self.lr_slider.setRange(1, 1000)
        self.lr_slider.setValue(30)
        self.lr_lbl = QLabel("0.030")
        self.lr_lbl.setFixedWidth(46)
        self.lr_slider.valueChanged.connect(self._lr_changed)
        lr_row.addWidget(self.lr_slider); lr_row.addWidget(self.lr_lbl)
        ctrl_lay.addLayout(lr_row)

        # Speed
        ctrl_lay.addWidget(self._lbl("Steps / frame"))
        sp_row = QHBoxLayout()
        self.sp_slider = QSlider(Qt.Orientation.Horizontal)
        self.sp_slider.setRange(1, 50); self.sp_slider.setValue(10)
        self.sp_lbl = QLabel("10")
        self.sp_lbl.setFixedWidth(30)
        self.sp_slider.valueChanged.connect(lambda v: self.sp_lbl.setText(str(v)))
        sp_row.addWidget(self.sp_slider); sp_row.addWidget(self.sp_lbl)
        ctrl_lay.addLayout(sp_row)

        self._sep(ctrl_lay)

        # Layer editor
        ctrl_lay.addWidget(self._lbl("LAYERS  (click + / − to resize)"))
        self.layer_editor = QWidget()
        self.layer_editor_lay = QHBoxLayout(self.layer_editor)
        self.layer_editor_lay.setSpacing(6)
        ctrl_lay.addWidget(self.layer_editor)
        self._refresh_layer_editor()

        # Add / remove layer
        add_rem = QHBoxLayout(); add_rem.setSpacing(8)
        btn_add = QPushButton("+ Add Layer")
        btn_rem = QPushButton("− Remove Layer")
        btn_add.setStyleSheet("background:#2b4a6f; border-radius:7px; padding:5px;")
        btn_rem.setStyleSheet("background:#4a2b2b; border-radius:7px; padding:5px;")
        btn_add.clicked.connect(self._add_layer)
        btn_rem.clicked.connect(self._remove_layer)
        add_rem.addWidget(btn_add); add_rem.addWidget(btn_rem)
        ctrl_lay.addLayout(add_rem)

        ctrl_lay.addStretch()
        main.addWidget(ctrl)

    def _lbl(self, txt):
        l = QLabel(txt)
        l.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        l.setStyleSheet("color:#505a72; letter-spacing:1px;")
        return l

    def _sep(self, layout):
        f = QFrame(); f.setFrameShape(QFrame.Shape.HLine)
        f.setStyleSheet("color:#232838;")
        layout.addWidget(f)

    # ── Layer editor ──────────────────────────────────────────────────────────
    def _refresh_layer_editor(self):
        # clear
        while self.layer_editor_lay.count():
            w = self.layer_editor_lay.takeAt(0).widget()
            if w: w.deleteLater()

        for i, size in enumerate(self.layer_sizes):
            box = QWidget()
            box.setStyleSheet("background:#1e2235; border-radius:8px;")
            vl  = QVBoxLayout(box); vl.setContentsMargins(4,4,4,4); vl.setSpacing(2)

            plus = QPushButton("+")
            plus.setFixedSize(28,22)
            plus.setStyleSheet("background:#276749; border-radius:5px; font-weight:bold; padding:0;")
            plus.clicked.connect(lambda _, idx=i: self._on_layer_changed(idx, 1))

            num = QLabel(str(size))
            num.setAlignment(Qt.AlignmentFlag.AlignCenter)
            num.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
            num.setStyleSheet("color:#c8d2e6;")

            minus = QPushButton("−")
            minus.setFixedSize(28,22)
            minus.setStyleSheet("background:#742a2a; border-radius:5px; font-weight:bold; padding:0;")
            minus.clicked.connect(lambda _, idx=i: self._on_layer_changed(idx, -1))

            vl.addWidget(plus,  alignment=Qt.AlignmentFlag.AlignHCenter)
            vl.addWidget(num,   alignment=Qt.AlignmentFlag.AlignHCenter)
            vl.addWidget(minus, alignment=Qt.AlignmentFlag.AlignHCenter)

            lname = QLabel("In" if i==0 else ("Out" if i==len(self.layer_sizes)-1 else f"H{i}"))
            lname.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lname.setStyleSheet("color:#505a72; font-size:9px;")
            vl.addWidget(lname)

            self.layer_editor_lay.addWidget(box)

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _on_layer_changed(self, idx, delta):
        self.layer_sizes[idx] = max(1, min(8, self.layer_sizes[idx] + delta))
        self._refresh_network()

    def _add_layer(self):
        if len(self.layer_sizes) < 7:
            self.layer_sizes.insert(-1, 4)
            self._refresh_network()

    def _remove_layer(self):
        if len(self.layer_sizes) > 1:
            self.layer_sizes.pop(-2)
            self._refresh_network()

    def _set_dataset(self, name):
        self.dataset_name = name
        self.dataset = DATASETS[name]()
        for n, b in self._ds_btns.items():
            b.setChecked(n == name)
        self._refresh_network()

    def _set_activation(self, name):
        self.activation = name
        for n, b in self._act_btns.items():
            b.setChecked(n == name)
        self._refresh_network()

    def _lr_changed(self, v):
        lr = v / 1000
        self.lr_lbl.setText(f"{lr:.3f}")
        if self.trainer:
            self.trainer.lr = lr

    def _get_lr(self):
        return self.lr_slider.value() / 1000

    def _get_speed(self):
        return self.sp_slider.value()

    def _refresh_network(self):
        self._pause_training()
        self.network  = build_network(self.layer_sizes, self.activation)
        self.loss_fn  = MSE()
        self.trainer  = None
        self.loss_graph.history.clear()
        self.epoch_lbl.setText("Epoch: 0")
        self.boundary.set_data(self.network, self.dataset)
        self.net_canvas.set_network(self.network, self.layer_sizes)
        self._refresh_layer_editor()
        self.boundary.recompute()

    def _start_training(self):
        if self.worker and self.worker.isRunning():
            return
        self.trainer = Trainer(self.network, self.loss_fn, self._get_lr())
        self.worker  = TrainWorker(self.trainer, self.dataset, self._get_speed)
        self.worker.step_done.connect(self._on_step)
        self.worker.start()

    def _pause_training(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None

    def _reset(self):
        self._refresh_network()

    def _on_step(self, loss, epoch):
        self.loss_graph.add(loss)
        self.epoch_lbl.setText(f"Epoch: {epoch}  |  Loss: {loss:.4f}")

    def _tick(self):
        if self.worker and self.worker.isRunning():
            self.boundary.recompute()
        self.net_canvas.update()

    def closeEvent(self, e):
        self._pause_training()
        e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
