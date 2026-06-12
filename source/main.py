import pygame
import numpy as np
import threading
import math
import sys
import os
import random

sys.path.insert(0, '/home/claude')
from neuron import NEURAL
from layer import LAYERS
from Network import Network
from losses import MSE
from BackPropagation import Trainer

# ─── COLORS ───────────────────────────────────────────────────────────────────
BG          = (15,  17,  26)
PANEL_BG    = (22,  25,  37)
ACCENT      = (99, 179, 237)
ACCENT2     = (154, 117, 234)
POS_COLOR   = (72, 199, 142)
NEG_COLOR   = (252, 110, 110)
NEURON_CLR  = (55,  65,  95)
NEURON_OUT  = (99, 179, 237)
TEXT_CLR    = (200, 210, 230)
DIM_CLR     = (80,  90, 110)
SLIDER_BG   = (35,  40,  58)
BTN_TRAIN   = (56, 161, 105)
BTN_PAUSE   = (214, 158,  46)
BTN_RESET   = (229,  62,  62)
WHITE       = (255, 255, 255)

# ─── LAYOUT ───────────────────────────────────────────────────────────────────
W, H        = 1280, 800
LEFT_W      = 380   # network visualizer
MID_W       = 420   # decision boundary
RIGHT_W     = W - LEFT_W - MID_W  # controls
MID_X       = LEFT_W
RIGHT_X     = LEFT_W + MID_W
GRID_RES    = 40    # decision boundary grid resolution

# ─── DATASETS ─────────────────────────────────────────────────────────────────
def make_xor():
    pts = []
    for _ in range(200):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        label = 1 if (x > 0) != (y > 0) else 0
        pts.append(([x, y], [label]))
    return pts

def make_circle():
    pts = []
    for _ in range(200):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        label = 1 if math.sqrt(x**2 + y**2) < 0.6 else 0
        pts.append(([x, y], [label]))
    return pts

def make_spiral():
    pts = []
    n = 100
    for i in range(n):
        t = i / n
        angle = t * 3 * math.pi
        r = t * 0.9
        x1 =  r * math.cos(angle) + random.gauss(0, 0.05)
        y1 =  r * math.sin(angle) + random.gauss(0, 0.05)
        x2 = -r * math.cos(angle) + random.gauss(0, 0.05)
        y2 = -r * math.sin(angle) + random.gauss(0, 0.05)
        pts.append(([x1, y1], [1]))
        pts.append(([x2, y2], [0]))
    return pts

DATASETS = {"XOR": make_xor, "Circle": make_circle, "Spiral": make_spiral}

# ─── NETWORK BUILDER ──────────────────────────────────────────────────────────
def build_network(layer_sizes, activation):
    layers = []
    for i, size in enumerate(layer_sizes):
        n_in = 2 if i == 0 else layer_sizes[i - 1]
        act  = "sigmoid" if i == len(layer_sizes) - 1 else activation
        neurons = [NEURAL(n_in, activation=act) for _ in range(size)]
        layers.append(LAYERS(neurons, n_in))
    return Network(layers)

# ─── SLIDER ───────────────────────────────────────────────────────────────────
class Slider:
    def __init__(self, x, y, w, mn, mx, val, label, fmt="{:.3f}", log=False):
        self.rect = pygame.Rect(x, y, w, 6)
        self.mn, self.mx, self.val = mn, mx, val
        self.label, self.fmt, self.log = label, fmt, log
        self.dragging = False
        self.handle_r = 8

    def _to_px(self):
        if self.log:
            t = (math.log10(self.val) - math.log10(self.mn)) / (math.log10(self.mx) - math.log10(self.mn))
        else:
            t = (self.val - self.mn) / (self.mx - self.mn)
        return int(self.rect.x + t * self.rect.w)

    def draw(self, surf, font_sm, font_med):
        pygame.draw.rect(surf, SLIDER_BG, self.rect, border_radius=3)
        hx = self._to_px()
        filled = pygame.Rect(self.rect.x, self.rect.y, hx - self.rect.x, 6)
        pygame.draw.rect(surf, ACCENT, filled, border_radius=3)
        pygame.draw.circle(surf, WHITE, (hx, self.rect.y + 3), self.handle_r)
        lbl = font_sm.render(self.label, True, DIM_CLR)
        surf.blit(lbl, (self.rect.x, self.rect.y - 18))
        val_s = font_sm.render(self.fmt.format(self.val), True, ACCENT)
        surf.blit(val_s, (self.rect.right - val_s.get_width(), self.rect.y - 18))

    def handle_event(self, e):
        hx = self._to_px()
        handle = pygame.Rect(hx - self.handle_r, self.rect.y + 3 - self.handle_r, self.handle_r*2, self.handle_r*2)
        if e.type == pygame.MOUSEBUTTONDOWN and handle.collidepoint(e.pos):
            self.dragging = True
        if e.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        if e.type == pygame.MOUSEMOTION and self.dragging:
            t = max(0, min(1, (e.pos[0] - self.rect.x) / self.rect.w))
            if self.log:
                self.val = self.mn * (self.mx / self.mn) ** t
            else:
                self.val = self.mn + t * (self.mx - self.mn)

# ─── BUTTON ───────────────────────────────────────────────────────────────────
class Button:
    def __init__(self, x, y, w, h, label, color):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.color = color
        self.hover = False

    def draw(self, surf, font):
        c = tuple(min(255, v + 20) for v in self.color) if self.hover else self.color
        pygame.draw.rect(surf, c, self.rect, border_radius=8)
        t = font.render(self.label, True, WHITE)
        surf.blit(t, t.get_rect(center=self.rect.center))

    def handle_event(self, e):
        self.hover = self.rect.collidepoint(pygame.mouse.get_pos())
        if e.type == pygame.MOUSEBUTTONDOWN and self.hover:
            return True
        return False

# ─── MAIN APP ─────────────────────────────────────────────────────────────────
class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Neural Network Playground")
        self.clock = pygame.time.Clock()

        self.font_lg  = pygame.font.SysFont("segoeui", 18, bold=True)
        self.font_med = pygame.font.SysFont("segoeui", 14)
        self.font_sm  = pygame.font.SysFont("segoeui", 12)

        # State
        self.layer_sizes   = [4, 4, 1]
        self.activation    = "sigmoid"
        self.dataset_name  = "XOR"
        self.dataset       = make_xor()
        self.training      = False
        self.epoch         = 0
        self.loss_history  = []
        self.lock          = threading.Lock()

        self.network = build_network(self.layer_sizes, self.activation)
        self.loss_fn = MSE()
        self.trainer = None

        # Decision boundary surface
        self.db_surface  = None
        self.db_dirty    = True

        # Controls (right panel)
        rx = RIGHT_X + 20
        self.lr_slider     = Slider(rx, 120, RIGHT_W-40, 0.001, 1.0, 0.03,  "Learning Rate", "{:.4f}", log=True)
        self.speed_slider  = Slider(rx, 185, RIGHT_W-40, 1,     50,  10,    "Steps / frame", "{:.0f}")

        btn_w = (RIGHT_W - 50) // 3
        self.btn_train = Button(rx,           260, btn_w, 36, "TRAIN",  BTN_TRAIN)
        self.btn_pause = Button(rx+btn_w+10,  260, btn_w, 36, "PAUSE",  BTN_PAUSE)
        self.btn_reset = Button(rx+2*(btn_w+10), 260, btn_w, 36, "RESET", BTN_RESET)

        # Dataset buttons
        self.ds_buttons = {}
        for i, name in enumerate(DATASETS):
            bx = rx + i * ((RIGHT_W - 40) // 3 + 3)
            self.ds_buttons[name] = Button(bx, 340, (RIGHT_W-46)//3, 30, name, (40,50,75))

        # Activation buttons
        self.act_names = ["sigmoid", "relu", "tanh", "leaky_relu"]
        self.act_buttons = {}
        for i, name in enumerate(self.act_names):
            bx = rx + (i % 2) * ((RIGHT_W-46)//2 + 3)
            by = 420 + (i // 2) * 38
            self.act_buttons[name] = Button(bx, by, (RIGHT_W-46)//2, 30, name, (40,50,75))

        # Neuron add/remove: track hovered layer
        self.hovered_layer = None

    # ── rebuild ───────────────────────────────────────────────────────────────
    def rebuild(self):
        with self.lock:
            self.training = False
            self.network  = build_network(self.layer_sizes, self.activation)
            self.loss_fn  = MSE()
            self.trainer  = None
            self.epoch    = 0
            self.loss_history = []
            self.db_dirty = True

    def reset_dataset(self):
        self.dataset = DATASETS[self.dataset_name]()
        self.rebuild()

    # ── training thread ───────────────────────────────────────────────────────
    def train_loop(self):
        steps = max(1, int(self.speed_slider.val))
        Xs = [d[0] for d in self.dataset]
        Ys = [d[1] for d in self.dataset]
        while self.training:
            self.trainer.lr = self.lr_slider.val
            loss = 0
            for _ in range(steps):
                loss = self.trainer.train_epoch(Xs, Ys)
            with self.lock:
                self.epoch += steps
                self.loss_history.append(loss)
                if len(self.loss_history) > 200:
                    self.loss_history.pop(0)
                self.db_dirty = True

    # ── decision boundary ─────────────────────────────────────────────────────
    def compute_db(self):
        size = min(MID_W, H) - 40
        surf = pygame.Surface((size, size))
        step = 2.0 / GRID_RES
        cell = size // GRID_RES
        with self.lock:
            for gx in range(GRID_RES):
                for gy in range(GRID_RES):
                    wx = -1 + gx * step + step/2
                    wy =  1 - gy * step - step/2
                    try:
                        val = self.network.predict([wx, wy])[0]
                    except:
                        val = 0.5
                    val = max(0, min(1, val))
                    r = int(NEG_COLOR[0]*(1-val) + POS_COLOR[0]*val)
                    g = int(NEG_COLOR[1]*(1-val) + POS_COLOR[1]*val)
                    b = int(NEG_COLOR[2]*(1-val) + POS_COLOR[2]*val)
                    pygame.draw.rect(surf, (r,g,b), (gx*cell, gy*cell, cell+1, cell+1))
        self.db_surface = surf
        self.db_dirty = False

    # ── draw network ──────────────────────────────────────────────────────────
    def draw_network(self):
        surf = self.screen
        # background panel
        pygame.draw.rect(surf, PANEL_BG, (0, 0, LEFT_W, H))
        title = self.font_lg.render("Network", True, TEXT_CLR)
        surf.blit(title, (15, 15))

        hint = self.font_sm.render("+ / -  on layer to add/remove neurons", True, DIM_CLR)
        surf.blit(hint, (15, H - 20))

        n_layers  = len(self.layer_sizes)
        margin_x  = 50
        spacing_x = (LEFT_W - 2*margin_x) / (n_layers - 0.5) if n_layers > 1 else LEFT_W/2
        area_y    = (50, H - 40)

        positions = []  # positions[layer][neuron] = (x,y)
        self.hovered_layer = None

        for li, size in enumerate(self.layer_sizes):
            cx   = int(margin_x + li * spacing_x)
            lpos = []
            for ni in range(size):
                t  = (ni + 1) / (size + 1)
                cy = int(area_y[0] + t * (area_y[1] - area_y[0]))
                lpos.append((cx, cy))
            positions.append(lpos)

            # layer hover zone
            lzone = pygame.Rect(cx - 30, area_y[0], 60, area_y[1]-area_y[0])
            if lzone.collidepoint(pygame.mouse.get_pos()):
                self.hovered_layer = li
                pygame.draw.rect(surf, (30, 35, 55), lzone, border_radius=10)
                # +/- hints
                plus  = self.font_lg.render("+", True, POS_COLOR)
                minus = self.font_lg.render("−", True, NEG_COLOR)
                surf.blit(plus,  (cx - plus.get_width()//2,  area_y[0] + 5))
                surf.blit(minus, (cx - minus.get_width()//2, area_y[1] - 25))

            # layer label
            lbl = "Out" if li == n_layers-1 else ("In" if li == 0 else f"H{li}")
            lt  = self.font_sm.render(lbl, True, DIM_CLR)
            surf.blit(lt, (cx - lt.get_width()//2, area_y[0] - 18))
            sz_t = self.font_sm.render(str(size), True, ACCENT)
            surf.blit(sz_t, (cx - sz_t.get_width()//2, area_y[0] - 4))

        # draw connections
        for li in range(len(positions) - 1):
            for (ax, ay) in positions[li]:
                for (bx, by) in positions[li+1]:
                    # get weight value if possible
                    try:
                        ni_b = positions[li+1].index((bx,by))
                        w    = self.network.layers[li+1].main_nodes[ni_b].weight[positions[li].index((ax,ay))]
                        w    = max(-1, min(1, w))
                        if w > 0:
                            c = (int(POS_COLOR[0]*abs(w)), int(POS_COLOR[1]*abs(w)), int(POS_COLOR[2]*abs(w)))
                        else:
                            c = (int(NEG_COLOR[0]*abs(w)), int(NEG_COLOR[1]*abs(w)), int(NEG_COLOR[2]*abs(w)))
                        thick = max(1, int(abs(w) * 3))
                    except:
                        c, thick = DIM_CLR, 1
                    pygame.draw.line(surf, c, (ax, ay), (bx, by), thick)

        # draw neurons
        for li, lpos in enumerate(positions):
            for ni, (cx, cy) in enumerate(lpos):
                try:
                    val = self.network.layers[li].main_nodes[ni].output
                    if val is None: val = 0.5
                    val = max(0, min(1, val))
                    fill = (int(NEURON_CLR[0] + val*(NEURON_OUT[0]-NEURON_CLR[0])),
                            int(NEURON_CLR[1] + val*(NEURON_OUT[1]-NEURON_CLR[1])),
                            int(NEURON_CLR[2] + val*(NEURON_OUT[2]-NEURON_CLR[2])))
                except:
                    fill = NEURON_CLR
                pygame.draw.circle(surf, fill, (cx, cy), 14)
                pygame.draw.circle(surf, NEURON_OUT, (cx, cy), 14, 2)

    # ── draw decision boundary ────────────────────────────────────────────────
    def draw_boundary(self):
        surf = self.screen
        pygame.draw.rect(surf, (18, 20, 32), (MID_X, 0, MID_W, H))
        title = self.font_lg.render("Decision Boundary", True, TEXT_CLR)
        surf.blit(title, (MID_X + 15, 15))

        size   = min(MID_W, H) - 80
        pad_x  = MID_X + (MID_W - size) // 2
        pad_y  = 45

        if self.db_surface:
            scaled = pygame.transform.scale(self.db_surface, (size, size))
            surf.blit(scaled, (pad_x, pad_y))
            pygame.draw.rect(surf, DIM_CLR, (pad_x, pad_y, size, size), 1)

        # plot data points
        for (xy, label) in self.dataset:
            px = int(pad_x + (xy[0] + 1) / 2 * size)
            py = int(pad_y + (1 - (xy[1] + 1) / 2) * size)
            c  = POS_COLOR if label[0] == 1 else NEG_COLOR
            pygame.draw.circle(surf, c,     (px, py), 5)
            pygame.draw.circle(surf, WHITE, (px, py), 5, 1)

        # epoch + loss
        ep_t = self.font_med.render(f"Epoch: {self.epoch}", True, TEXT_CLR)
        surf.blit(ep_t, (MID_X + 15, H - 50))
        if self.loss_history:
            ls_t = self.font_med.render(f"Loss: {self.loss_history[-1]:.4f}", True, ACCENT)
            surf.blit(ls_t, (MID_X + 15, H - 30))

        # mini loss graph
        if len(self.loss_history) > 1:
            gx, gy, gw, gh = MID_X + 15, H - 130, MID_W - 30, 70
            pygame.draw.rect(surf, PANEL_BG, (gx, gy, gw, gh), border_radius=4)
            mx = max(self.loss_history) or 1
            pts = []
            for i, l in enumerate(self.loss_history):
                px2 = gx + int(i / max(1,len(self.loss_history)-1) * gw)
                py2 = gy + gh - int(l / mx * gh)
                pts.append((px2, py2))
            if len(pts) > 1:
                pygame.draw.lines(surf, ACCENT2, False, pts, 2)
            lg = self.font_sm.render("Loss", True, DIM_CLR)
            surf.blit(lg, (gx+4, gy+4))

    # ── draw controls ─────────────────────────────────────────────────────────
    def draw_controls(self):
        surf = self.screen
        pygame.draw.rect(surf, PANEL_BG, (RIGHT_X, 0, RIGHT_W, H))
        title = self.font_lg.render("Controls", True, TEXT_CLR)
        surf.blit(title, (RIGHT_X + 20, 15))

        # layer editor
        sec = self.font_med.render("LAYERS", True, DIM_CLR)
        surf.blit(sec, (RIGHT_X + 20, 55))

        rx    = RIGHT_X + 20
        lw    = RIGHT_W - 40
        btn_h = 28
        max_l = 6
        ls    = self.layer_sizes
        cell  = lw // max_l

        for i in range(max_l):
            bx = rx + i * cell
            by = 72
            if i < len(ls):
                # existing layer — show size, click to remove
                pygame.draw.rect(surf, ACCENT2, (bx+2, by, cell-4, btn_h), border_radius=6)
                t = self.font_sm.render(str(ls[i]), True, WHITE)
                surf.blit(t, t.get_rect(center=(bx + cell//2, by + btn_h//2)))
            else:
                # empty slot — click to add
                pygame.draw.rect(surf, SLIDER_BG, (bx+2, by, cell-4, btn_h), border_radius=6)
                t = self.font_sm.render("+", True, DIM_CLR)
                surf.blit(t, t.get_rect(center=(bx + cell//2, by + btn_h//2)))

        self.layer_rects = [pygame.Rect(rx + i*cell+2, 72, cell-4, btn_h) for i in range(max_l)]

        # sliders
        self.lr_slider.draw(surf, self.font_sm, self.font_med)
        self.speed_slider.draw(surf, self.font_sm, self.font_med)

        # buttons
        self.btn_train.draw(surf, self.font_med)
        self.btn_pause.draw(surf, self.font_med)
        self.btn_reset.draw(surf, self.font_med)

        # status
        status = "● TRAINING" if self.training else "■ PAUSED"
        sc = POS_COLOR if self.training else DIM_CLR
        st = self.font_sm.render(status, True, sc)
        surf.blit(st, (rx, 304))

        # dataset
        ds_lbl = self.font_med.render("DATASET", True, DIM_CLR)
        surf.blit(ds_lbl, (rx, 320))
        for name, btn in self.ds_buttons.items():
            btn.color = ACCENT if name == self.dataset_name else (40, 50, 75)
            btn.draw(surf, self.font_sm)

        # activation
        act_lbl = self.font_med.render("ACTIVATION (hidden)", True, DIM_CLR)
        surf.blit(act_lbl, (rx, 400))
        for name, btn in self.act_buttons.items():
            btn.color = ACCENT2 if name == self.activation else (40, 50, 75)
            btn.draw(surf, self.font_sm)

        # layer size hint
        hint = self.font_sm.render("Click layer box to change size", True, DIM_CLR)
        surf.blit(hint, (rx, 510))

        # layer size +/- for selected layer
        ls_lbl = self.font_med.render("NEURONS PER LAYER", True, DIM_CLR)
        surf.blit(ls_lbl, (rx, 530))
        for i, size in enumerate(self.layer_sizes):
            bx = rx + i * (lw // len(self.layer_sizes))
            by = 550
            bw = lw // len(self.layer_sizes) - 4

            pygame.draw.rect(surf, SLIDER_BG, (bx, by, bw, 60), border_radius=6)
            num = self.font_lg.render(str(size), True, TEXT_CLR)
            surf.blit(num, num.get_rect(center=(bx+bw//2, by+30)))

            plus_r  = pygame.Rect(bx+bw-18, by+4,  14, 14)
            minus_r = pygame.Rect(bx+4,     by+4,  14, 14)
            pygame.draw.rect(surf, POS_COLOR, plus_r,  border_radius=3)
            pygame.draw.rect(surf, NEG_COLOR, minus_r, border_radius=3)
            p = self.font_sm.render("+", True, WHITE)
            m = self.font_sm.render("−", True, WHITE)
            surf.blit(p, p.get_rect(center=plus_r.center))
            surf.blit(m, m.get_rect(center=minus_r.center))

            self._plus_rects  = getattr(self, '_plus_rects',  [])
            self._minus_rects = getattr(self, '_minus_rects', [])
            if len(self._plus_rects)  <= i: self._plus_rects.append(plus_r)
            else: self._plus_rects[i]  = plus_r
            if len(self._minus_rects) <= i: self._minus_rects.append(minus_r)
            else: self._minus_rects[i] = minus_r

    # ── events ────────────────────────────────────────────────────────────────
    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.training = False
                pygame.quit()
                sys.exit()

            self.lr_slider.handle_event(e)
            self.speed_slider.handle_event(e)

            if self.btn_train.handle_event(e):
                if not self.training:
                    self.training = True
                    self.trainer  = Trainer(self.network, self.loss_fn, self.lr_slider.val)
                    t = threading.Thread(target=self.train_loop, daemon=True)
                    t.start()

            if self.btn_pause.handle_event(e):
                self.training = False

            if self.btn_reset.handle_event(e):
                self.rebuild()

            for name, btn in self.ds_buttons.items():
                if btn.handle_event(e):
                    self.dataset_name = name
                    self.reset_dataset()

            for name, btn in self.act_buttons.items():
                if btn.handle_event(e):
                    self.activation = name
                    self.rebuild()

            # layer neuron +/-
            if e.type == pygame.MOUSEBUTTONDOWN:
                for i in range(len(self.layer_sizes)):
                    if i < len(getattr(self, '_plus_rects', [])):
                        if self._plus_rects[i].collidepoint(e.pos):
                            self.layer_sizes[i] = min(8, self.layer_sizes[i] + 1)
                            self.rebuild()
                        if self._minus_rects[i].collidepoint(e.pos):
                            self.layer_sizes[i] = max(1, self.layer_sizes[i] - 1)
                            self.rebuild()

            # add/remove layers via layer panel rects
            if e.type == pygame.MOUSEBUTTONDOWN:
                if hasattr(self, 'layer_rects'):
                    for i, r in enumerate(self.layer_rects):
                        if r.collidepoint(e.pos):
                            if i < len(self.layer_sizes):
                                if len(self.layer_sizes) > 1:
                                    self.layer_sizes.pop(i)
                                    self.rebuild()
                            else:
                                if i == len(self.layer_sizes) and len(self.layer_sizes) < 6:
                                    self.layer_sizes.insert(-1, 4)
                                    self.rebuild()

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self):
        # initial boundary
        self.compute_db()

        while True:
            self.handle_events()

            if self.db_dirty:
                threading.Thread(target=self.compute_db, daemon=True).start()
                self.db_dirty = False

            self.screen.fill(BG)
            self.draw_network()
            self.draw_boundary()
            self.draw_controls()

            # dividers
            pygame.draw.line(self.screen, (35,40,58), (MID_X, 0), (MID_X, H), 2)
            pygame.draw.line(self.screen, (35,40,58), (RIGHT_X, 0), (RIGHT_X, H), 2)

            pygame.display.flip()
            self.clock.tick(30)

if __name__ == "__main__":
    App().run()
