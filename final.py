import sys
import os
import time
import random
from glob import glob
from collections import defaultdict

import cv2
import numpy as np
import serial
from ultralytics import YOLO

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt

# ---------------- SETTINGS ----------------
WEIGHTS = r"runs/classify/train/weights/best.pt"
TEST_FOLDER = r"C:\Users\DELL Latitude 7400\Desktop\smart_waste\waste_split\val"

# --- SERIAL CONFIGURATION ---
SERIAL_PORT = 'COM7'  
BAUD_RATE = 9600
# ------------------------------------

# --- LAG SETTING (IMPORTANT) ---
# Time to freeze Python while Proteus moves (in milliseconds)
PROTEUS_LAG_TIME = 6000 
# -------------------------------

TOPK = 1
CONF_THRESHOLD = 0.20
CANVAS_W, CANVAS_H = 1280, 720
BACKGROUND = (230, 230, 230)
AUTO_ADVANCE_SECS = 2.0 

BIN_NAMES = ["BLUE (Dry & Recyclable)", "RED (Hazardous)",
             "GREEN (Organic)", "BLACK (General)"]
BIN_COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (40, 40, 40)]

CLASS_TO_BIN = {
    # BLUE
    "paper":       (0, "BLUE"),
    "cardboard":   (0, "BLUE"),
    "plastic":     (0, "BLUE"),
    "metal":       (0, "BLUE"),
    "brown-glass": (0, "BLUE"),
    "green-glass": (0, "BLUE"),
    "white-glass": (0, "BLUE"),
    # RED
    "battery":     (1, "RED"),
    # GREEN
    "biological":  (2, "GREEN"),
    # BLACK
    "trash":       (3, "BLACK"),
    "clothes":     (3, "BLACK"),
    "shoes":       (3, "BLACK"),
}

# ---------- helpers ----------
def letterbox_to_canvas(img, out_w=CANVAS_W, out_h=CANVAS_H, color=BACKGROUND):
    if img is None:
        return np.full((out_h, out_w, 3), color, dtype=np.uint8)
    h, w = img.shape[:2]
    r = min(out_w / w, out_h / h)
    nw, nh = int(w * r), int(h * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((out_h, out_w, 3), color, dtype=np.uint8)
    x0 = (out_w - nw) // 2
    y0 = (out_h - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def font_params(base_scale=1.0, base_th=2):
    s = (CANVAS_H / 720.0) * base_scale
    th = max(1, int((CANVAS_H / 720.0) * base_th))
    return s, th

def draw_bins(frame, state):
    h, w = frame.shape[:2]
    
    # Layout calculations
    left_pane_w = int(0.33 * w)
    outer_margin_r = max(24, int(0.04 * w))
    inner_gap = max(16, int(0.02 * w))
    
    x_start = left_pane_w + max(16, int(0.02 * w))
    x_end = w - outer_margin_r
    usable_w = max(200, x_end - x_start)
    
    bin_w = max(40, (usable_w - 3 * inner_gap) // 4)
    bin_h = max(80, int(h * 0.55))
    top = int(h * 0.35)

    for i in range(4):
        x = x_start + i * (bin_w + inner_gap)
        color = BIN_COLORS[i]

        # 1. Draw Bin Body
        cv2.rectangle(frame, (x, top), (x + bin_w, top + bin_h), color, 3)

        # 2. Draw Lid
        lid_th = max(8, int(0.02 * h))
        open_px = int(state[i] * 60)
        lid_y = top - open_px
        cv2.rectangle(frame, (x, lid_y - lid_th), (x + bin_w, lid_y), color, -1)

        # 3. DRAW TEXT (Split into 2 lines to fix overlapping)
        full_text = BIN_NAMES[i]
        
        if " (" in full_text:
            parts = full_text.split(" (")
            main_text = parts[0]       # e.g., "BLUE"
            sub_text = "(" + parts[1]  # e.g., "(Dry & Recyclable)"
        else:
            main_text = full_text
            sub_text = ""

        # Draw Color Name (Big Font)
        cv2.putText(frame, main_text, 
                    (x + 5, top + bin_h + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        # Draw Description (Small Font)
        cv2.putText(frame, sub_text, 
                    (x + 5, top + bin_h + 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

# -------------- PyQt6 GUI --------------
class SmartBinWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Bin Simulator (4 bins)")
        self.resize(CANVAS_W // 2, CANVAS_H // 2)

        # --- SERIAL CONNECTION INIT ---
        self.ser = None
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            print(f"✅ Connected to Proteus on {SERIAL_PORT}")
        except Exception as e:
            print(f"❌ Could not connect to {SERIAL_PORT}. Error: {e}")
            print("   (Hardware simulation will not work, but GUI will run)")
        # -----------------------------

        self.model = YOLO(WEIGHTS)
        self.imgs = []
        for ext in ("**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.bmp", "**/*.webp"):
            self.imgs.extend(glob(os.path.join(TEST_FOLDER, ext), recursive=True))
        
        if not self.imgs:
            print(f"❌ No images found under {TEST_FOLDER}")
        else:
            # --- NEW METHOD: SHUFFLED BATCHES (Natural Balance) ---
            
            # 1. Group images by Bin Color
            buckets = defaultdict(list)
            for img_path in self.imgs:
                folder_name = os.path.basename(os.path.dirname(img_path))
                if folder_name in CLASS_TO_BIN:
                    _, bin_name = CLASS_TO_BIN[folder_name]
                    buckets[bin_name].append(img_path)
            
            # Shuffle inside the buckets first (so we don't pick the same apple twice)
            for key in buckets:
                random.shuffle(buckets[key])

            balanced_imgs = []
            keys = list(buckets.keys()) 

            # 2. Create small batches until data runs out
            while True:
                batch = []
                empty_buckets = 0
                
                # Try to pick one image from EVERY color
                for key in keys:
                    if len(buckets[key]) > 0:
                        batch.append(buckets[key].pop(0))
                    else:
                        empty_buckets += 1
                
                # If all buckets are empty, stop
                if len(batch) == 0:
                    break
                
                # 3. SHUFFLE THE BATCH (The Magic Step)
                # This makes it look random, not robotic
                random.shuffle(batch)
                
                balanced_imgs.extend(batch)

            self.imgs = balanced_imgs
            # ------------------------------------------------------
            
        self.idx = 0

        self.counts = defaultdict(int)
        self.lid_state = [0.0] * 4
        self.last_label = "—"
        self.last_conf = 0.0
        self.last_bin_name = "None"

        # --- ANIMATION SETUP ---
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self.run_animation_frame)
        self.anim_phase = 0       # 0=Idle, 1=Opening, 2=Holding, 3=Closing
        self.anim_target_idx = -1 # Which bin is animating
        self.anim_angle = 0.0     # 0.0 to 1.0
        self.anim_hold_counter = 0
        self.current_frame_static = None # Holds the image steady while animating
        # -----------------------

        self.canvas_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.canvas_label.setStyleSheet("background-color: black;")

        self.btn_blue = QPushButton("BLUE")
        self.btn_red = QPushButton("RED")
        self.btn_green = QPushButton("GREEN")
        self.btn_black = QPushButton("BLACK")
        self.btn_close_all = QPushButton("CLOSE ALL")
        self.btn_auto = QPushButton("AUTO DEMO")

        for b in (self.btn_blue, self.btn_red, self.btn_green, self.btn_black,
                  self.btn_close_all, self.btn_auto):
            b.setMinimumHeight(40)

        self.btn_blue.clicked.connect(lambda: self.manual_command("BLUE"))
        self.btn_red.clicked.connect(lambda: self.manual_command("RED"))
        self.btn_green.clicked.connect(lambda: self.manual_command("GREEN"))
        self.btn_black.clicked.connect(lambda: self.manual_command("BLACK"))
        self.btn_close_all.clicked.connect(self.close_all_lids)
        self.btn_auto.clicked.connect(self.toggle_auto)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_blue)
        btn_layout.addWidget(self.btn_red)
        btn_layout.addWidget(self.btn_green)
        btn_layout.addWidget(self.btn_black)
        btn_layout.addWidget(self.btn_close_all)
        btn_layout.addWidget(self.btn_auto)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.canvas_label, stretch=1)
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

        self.auto_on = False
        self.timer = QTimer(self)
        self.timer.setInterval(int(AUTO_ADVANCE_SECS * 1000))
        self.timer.timeout.connect(self.process_next_image)

        self.update_canvas(None, None)

    def run_animation_frame(self):
        """Updates lid angle smoothly: Open -> Hold -> Close"""
        # PHASE 1: OPENING
        if self.anim_phase == 1:
            self.anim_angle += 0.1  # Speed (increase to make faster)
            if self.anim_angle >= 1.0:
                self.anim_angle = 1.0
                self.anim_phase = 2 # Switch to Hold
                self.anim_hold_counter = 0

        # PHASE 2: HOLDING
        elif self.anim_phase == 2:
            self.anim_hold_counter += 1
            if self.anim_hold_counter > 20: # Hold time (approx 0.6s)
                self.anim_phase = 3 # Switch to Close

        # PHASE 3: CLOSING
        elif self.anim_phase == 3:
            self.anim_angle -= 0.1
            if self.anim_angle <= 0.0:
                self.anim_angle = 0.0
                self.anim_phase = 0 # Done
                self.anim_timer.stop()

        # Update the state list so draw_bins sees the new angle
        if self.anim_target_idx != -1:
            self.lid_state = [0.0] * 4 # Close others
            self.lid_state[self.anim_target_idx] = self.anim_angle

        # Redraw the screen (Pass None so it doesn't reset our manual state)
        if self.current_frame_static is not None:
            self.update_canvas(self.current_frame_static, None)

    def send_serial_command(self, bin_name):
        """Sends a single character to Proteus via Serial."""
        if self.ser and self.ser.is_open:
            # Protocol: B=Blue, R=Red, G=Green, K=Black, X=Close
            cmd = b'X'
            if bin_name == "BLUE": cmd = b'B'
            elif bin_name == "RED": cmd = b'R'
            elif bin_name == "GREEN": cmd = b'G'
            elif bin_name == "BLACK": cmd = b'K'
            elif bin_name == "CLOSE": cmd = b'X'
            
            try:
                self.ser.write(cmd)
                print(f"Sent Serial: {cmd}")
            except Exception as e:
                print(f"Serial Write Error: {e}")

    def process_next_image(self):
        if not self.imgs: return
        
        path = self.imgs[self.idx]
        self.idx = (self.idx + 1) % len(self.imgs)
        raw = cv2.imread(path)
        if raw is None: return

        panel = letterbox_to_canvas(raw, CANVAS_W, CANVAS_H, BACKGROUND)
        pred = self.model.predict(source=panel, verbose=False)[0]

        if pred.probs is not None:
            probs = pred.probs.data.cpu().numpy()
            top_idx = probs.argmax()
            self.last_conf = float(probs[top_idx])
            self.last_label = self.model.names[top_idx]
        else:
            self.last_label = "unknown"
            self.last_conf = 0.0

        chosen = None
        bin_triggered = False

        # Only process if we are NOT currently animating
        if self.anim_phase == 0:
            if self.last_conf >= CONF_THRESHOLD and self.last_label in CLASS_TO_BIN:
                bin_idx, bin_name = CLASS_TO_BIN[self.last_label]
                chosen = (bin_idx, bin_name)
                self.counts[bin_name] += 1
                
                # 1. SEND TO HARDWARE
                self.send_serial_command(bin_name)
                
                # 2. START VISUAL ANIMATION
                self.current_frame_static = panel.copy() # Freeze image
                self.anim_target_idx = bin_idx
                self.anim_angle = 0.0
                self.anim_phase = 1 # Start Opening
                self.anim_timer.start(30) # 30ms per frame
                
                bin_triggered = True
            else:
                pass

        # Update canvas initially
        self.update_canvas(panel, chosen)

        # 3. FREEZE MAIN LOGIC (Prevent Traffic Jam)
        if bin_triggered and self.auto_on:
            self.timer.stop() # Stop detecting new images
            # Wait for Proteus, then resume
            QTimer.singleShot(PROTEUS_LAG_TIME, self.resume_auto_demo)

    def resume_auto_demo(self):
        """Called after the wait time is over."""
        if self.auto_on:
            self.timer.start()
            self.process_next_image()

    def manual_command(self, bin_name):
        mapping = {"BLUE": 0, "RED": 1, "GREEN": 2, "BLACK": 3}
        bin_idx = mapping.get(bin_name)
        if bin_idx is None: return
        self.last_label = f"manual-{bin_name.lower()}"
        self.last_conf = 1.0
        self.counts[bin_name] += 1
        
        self.send_serial_command(bin_name)
        self.update_canvas(None, (bin_idx, bin_name))

    def close_all_lids(self):
        self.lid_state = [0.0] * 4
        self.send_serial_command("CLOSE")
        self.update_canvas(None, None)

    def toggle_auto(self):
        self.auto_on = not self.auto_on
        if self.auto_on:
            self.btn_auto.setText("AUTO DEMO (ON)")
            self.timer.start()
        else:
            self.btn_auto.setText("AUTO DEMO")
            self.timer.stop()

    def update_canvas(self, base_img, chosen):
        if base_img is None:
            panel = np.full((CANVAS_H, CANVAS_W, 3), BACKGROUND, dtype=np.uint8)
        else:
            panel = base_img.copy()

        if chosen is not None:
            bin_idx, bin_name = chosen
            self.last_bin_name = bin_name
            # ONLY snap to 1.0 if we are NOT running the smooth animation
            if self.anim_phase == 0: 
                for i in range(4):
                    self.lid_state[i] = 1.0 if i == bin_idx else 0.0
        
        draw_bins(panel, self.lid_state)

        title_scale, title_th = font_params(base_scale=1.4, base_th=3)
        line_scale, line_th = font_params(base_scale=1.0, base_th=3)
        count_scale, count_th = font_params(base_scale=1.2, base_th=3)

        cv2.putText(panel, f"Type of Waste: {self.last_label} ({self.last_conf:.2f})",
                    (24, 60), cv2.FONT_HERSHEY_SIMPLEX, title_scale, (0, 0, 0), title_th, cv2.LINE_AA)
        cv2.putText(panel, f"{self.last_label}  -->  {self.last_bin_name}",
                    (24, 120), cv2.FONT_HERSHEY_SIMPLEX, line_scale, (0, 0, 0), line_th, cv2.LINE_AA)

        y = 180
        for name in ["BLUE", "RED", "GREEN", "BLACK"]:
            cv2.putText(panel, f"{name}: {self.counts[name]}",
                        (24, y), cv2.FONT_HERSHEY_SIMPLEX, count_scale, (0, 0, 0), count_th, cv2.LINE_AA)
            y += int(44 * (CANVAS_H / 720.0))

        rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.canvas_label.setPixmap(pix.scaled(
            self.canvas_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

def main():
    app = QApplication(sys.argv)
    win = SmartBinWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()