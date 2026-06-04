#!/usr/bin/env python3
"""
sih_scan_final.py - end-to-end scanning, ML inference, spray decision, and upload.

Usage:
  python3 sih_scan_final.py
  python3 sih_scan_final.py --mode upload --upload-paths "/tmp/a.jpg,/tmp/b.jpg,/tmp/c.jpg"
  python3 sih_scan_final.py --do-spray
  python3 sih_scan_final.py --dry-run-upload
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime
from shutil import which as shutil_which, copy as shutil_copy

import cv2
import numpy as np

# ---------- TFLite interpreter preference ----------
try:
    from tflite_runtime.interpreter import Interpreter
    TFLITE_IMPL = "tflite_runtime"
except Exception:
    try:
        from tensorflow.lite import Interpreter
        TFLITE_IMPL = "tensorflow"
    except Exception:
        Interpreter = None
        TFLITE_IMPL = None

# ---------- gpiozero / pigpio detection ----------
PIGS_AVAILABLE = shutil_which("pigs") is not None
PiGPIOFactory = None
Servo = None
try:
    from gpiozero import OutputDevice, Servo
    try:
        from gpiozero.pins.pigpio import PiGPIOFactory as _PiGPIOFactory
        PiGPIOFactory = _PiGPIOFactory
    except Exception:
        PiGPIOFactory = None
    GPIOZERO_AVAILABLE = True
except Exception:
    OutputDevice = None
    GPIOZERO_AVAILABLE = False

# ---------- Config ----------
USER_HOME = Path("/home/sih")
SIH_DIR = USER_HOME / "SIH"
IMG_DIR = SIH_DIR / "images"
COUNTER_FILE = SIH_DIR / "counters.json"
LOG_FILE = SIH_DIR / "sih_scan_final.log"
LABELS_FILE = SIH_DIR / "labels.json"
MODEL_PATH = SIH_DIR / "model.tflite"

RCLONE_BASE = "sih1drive:SIH_Images"
RCLONE_ORIG = f"{RCLONE_BASE}/original"
RCLONE_PROC = f"{RCLONE_BASE}/processed"

CAM_CMD_CANDIDATES = ["libcamera-still", "rpicam-still", "raspistill"]

# BCM pins
SERVO_GPIO = 13
PUMP_GPIO = 27

CAPTURE_RES = (2000, 1500)
CAPTURE_TIMEOUT = 15
SERVO_SETTLE_SEC = 1.0

BLUE_THRESHOLD = 110
STRESS_THRESHOLD_PERCENT = 15.0

PROCESS_SCALE = 0.45
MIN_PROC_WIDTH = 800

MAX_SPRAY_SEC = 10.0
DEFAULT_SPRAY_LOW_SEC = 1.5
DEFAULT_SPRAY_HIGH_SEC = 3.5
SPRAY_PROBABILITY_MIN = 0.6

pos_to_prefix = {"left":"l", "center":"c", "right":"r"}

# ---------- logging ----------
def log(msg):
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t}] {msg}"
    print(line)
    try:
        SIH_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass

# ---------- dirs and counters ----------
def ensure_dirs_and_counters():
    SIH_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    if not COUNTER_FILE.exists():
        COUNTER_FILE.write_text(json.dumps({"r":0,"l":0,"c":0}))
    log("Ensured dirs and counters")

def get_and_inc(prefix):
    data = json.loads(COUNTER_FILE.read_text())
    if prefix not in data:
        data[prefix] = 0
    data[prefix] += 1
    tmp = COUNTER_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(COUNTER_FILE)
    return data[prefix]

# ---------- camera ----------
def find_camera_cmd():
    for c in CAM_CMD_CANDIDATES:
        if shutil_which(c):
            return c
    return None

def take_photo(dest_path: Path):
    camexe = find_camera_cmd()
    if not camexe:
        log("No camera binary found.")
        return False

    if "libcamera-still" in camexe:
        cmd = [camexe, "-n", "-o", str(dest_path), "-t", "2000",
               "--width", str(CAPTURE_RES[0]), "--height", str(CAPTURE_RES[1])]
    elif "rpicam-still" in camexe:
        cmd = [camexe, "-t", "2000", "-o", str(dest_path)]
    else:
        cmd = ["raspistill", "-o", str(dest_path),
               "-w", str(CAPTURE_RES[0]), "-h", str(CAPTURE_RES[1]), "-n", "-t", "2000"]

    log(f"Capturing: {' '.join(cmd)}")
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=CAPTURE_TIMEOUT)
        if p.returncode != 0:
            log("Camera error: " + p.stderr.decode().strip())
            return dest_path.exists()
        time.sleep(0.1)
        return dest_path.exists()
    except Exception as e:
        log("Capture exception: " + str(e))
        return False

# ---------- TFLite ----------
def load_tflite_model(model_path: Path):
    if Interpreter is None:
        log("Interpreter not available")
        return None
    if not model_path.exists():
        log(f"TFLite model not found at {model_path}")
        return None
    try:
        interp = Interpreter(str(model_path))
        interp.allocate_tensors()
        log(f"TFLite model loaded ({TFLITE_IMPL}).")
        return interp
    except Exception as e:
        log(f"Failed to load tflite model: {e}")
        return None

def run_tflite_on_crop(interp, img_bgr):
    if interp is None:
        return {"error": "interpreter_missing"}
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    in_det = input_details[0]
    in_shape = in_det["shape"]
    if len(in_shape) == 4:
        target_h = int(in_shape[1]); target_w = int(in_shape[2])
    elif len(in_shape) == 3:
        target_h = int(in_shape[0]); target_w = int(in_shape[1])
    else:
        target_h, target_w = 224, 224

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    input_dtype = in_det.get("dtype", np.float32)
    if np.issubdtype(input_dtype, np.floating):
        input_data = img_resized.astype(np.float32) / 255.0
    else:
        input_data = img_resized.astype(input_dtype)
    if len(input_data.shape) == 3:
        input_data = np.expand_dims(input_data, 0)

    try:
        interp.set_tensor(in_det["index"], input_data.astype(in_det["dtype"]))
        interp.invoke()
    except Exception as e:
        log(f"Model invocation failed: {e}")
        return {"error": f"invoke_failed: {e}"}

    outputs = {}
    try:
        for out in output_details:
            out_idx = out["index"]
            out_val = interp.get_tensor(out_idx)
            outputs[str(out_idx)] = out_val
    except Exception as e:
        log(f"Failed to read outputs: {e}")
        return {"error": f"output_read_failed: {e}"}

    structured = interpret_model_output(outputs)
    outputs_serial = {k: v.tolist() for k, v in outputs.items()}
    return {"raw_outputs": outputs_serial, "structured": structured}

def interpret_model_output(outputs_dict):
    structured = {
        "class": None,
        "class_confidence": None,
        "spray_probability": None,
        "spray_seconds": None,
        "severity": None,
        "spray_decision": None,
        "notes": []
    }
    labels_map = {}
    try:
        if LABELS_FILE.exists():
            labels_map = json.loads(LABELS_FILE.read_text())
            structured["notes"].append("labels_loaded")
    except Exception as e:
        structured["notes"].append(f"labels_load_failed:{e}")

    for name, arr in outputs_dict.items():
        a = np.array(arr).squeeze()
        if a.size == 0:
            continue
        # classification vector
        if a.ndim == 1 and a.size >= 2:
            s = float(np.sum(a))
            if s > 0.9 and s < 1.1 and np.all(a >= -1e-6):
                idx = int(np.argmax(a))
                conf = float(a[idx])
                structured["class"] = labels_map.get(str(idx), {}).get("label", f"class_{idx}")
                structured["class_confidence"] = conf
                structured["notes"].append(f"class_prob_vector_tensor:{name}")
                lbl_info = labels_map.get(str(idx), {})
                if isinstance(lbl_info, dict) and "spray" in lbl_info:
                    spray_pref = lbl_info.get("spray")
                    if spray_pref in ("none", "no"):
                        structured["spray_decision"] = False
                    elif spray_pref == "low":
                        structured["spray_decision"] = True
                        structured["spray_seconds"] = DEFAULT_SPRAY_LOW_SEC
                    elif spray_pref == "high":
                        structured["spray_decision"] = True
                        structured["spray_seconds"] = DEFAULT_SPRAY_HIGH_SEC
                    structured["notes"].append(f"spray_pref_from_labels:{spray_pref}")
                continue
        # scalar output
        if np.isscalar(a) or (a.ndim == 0) or (a.ndim == 1 and a.size == 1):
            v = float(np.array(a).reshape(-1)[0])
            if 0.0 <= v <= 1.0:
                if structured["spray_probability"] is None:
                    structured["spray_probability"] = v
                    structured["notes"].append(f"spray_prob_scalar_tensor:{name}")
                else:
                    if structured["severity"] is None:
                        structured["severity"] = v
                        structured["notes"].append(f"severity_scalar_tensor:{name}")
                    else:
                        structured["notes"].append(f"extra_scalar_tensor:{name}")
                continue
            if 0.0 < v <= MAX_SPRAY_SEC * 2.0:
                if structured["spray_seconds"] is None:
                    structured["spray_seconds"] = float(v)
                    structured["notes"].append(f"spray_seconds_tensor:{name}")
                    continue
            if structured["severity"] is None:
                structured["severity"] = float(v)
                structured["notes"].append(f"severity_from_tensor:{name}")
                continue
        # matrix-like outputs -> try flatten classification
        if a.ndim >= 2:
            try:
                flat = a.flatten()
                if flat.size > 0:
                    s = float(np.sum(flat))
                    if s > 0.9 and s < 1.1 and np.all(flat >= -1e-6):
                        idx = int(np.argmax(flat))
                        conf = float(flat[idx])
                        structured["class"] = labels_map.get(str(idx), {}).get("label", f"class_{idx}")
                        structured["class_confidence"] = conf
                        structured["notes"].append(f"class_prob_flat_tensor:{name}")
                        continue
                    v = float(flat[0])
                    if structured["severity"] is None:
                        structured["severity"] = v
                        structured["notes"].append(f"severity_from_flat_tensor:{name}")
            except Exception:
                pass

    # decide spray by probability if provided
    if structured.get("spray_probability") is not None:
        if structured["spray_probability"] >= SPRAY_PROBABILITY_MIN:
            if structured.get("spray_seconds") is None:
                if structured.get("severity") is not None and structured["severity"] >= 0.6:
                    structured["spray_seconds"] = DEFAULT_SPRAY_HIGH_SEC
                else:
                    structured["spray_seconds"] = DEFAULT_SPRAY_LOW_SEC
            structured["spray_decision"] = True
            structured["notes"].append("spray_decision_from_prob")
        else:
            structured["spray_decision"] = False
            structured["notes"].append("prob_below_threshold")

    if structured.get("class") is not None:
        c = structured["class"].lower()
        if "healthy" in c or "none" in c:
            structured["spray_decision"] = False
            structured["spray_seconds"] = 0.0
            structured["notes"].append("class_suggests_no_spray")

    if structured.get("spray_decision") is None:
        if structured.get("severity") is not None:
            s = structured["severity"]
            if s >= 0.6:
                structured["spray_decision"] = True
                structured["spray_seconds"] = DEFAULT_SPRAY_HIGH_SEC
                structured["notes"].append("spray_from_severity_high")
            elif s >= 0.3:
                structured["spray_decision"] = True
                structured["spray_seconds"] = DEFAULT_SPRAY_LOW_SEC
                structured["notes"].append("spray_from_severity_med")
            else:
                structured["spray_decision"] = False
                structured["notes"].append("severity_low_no_spray")
        else:
            structured["spray_decision"] = False
            structured["notes"].append("no_clear_signal_no_spray")

    if structured.get("spray_seconds") is not None:
        structured["spray_seconds"] = float(min(MAX_SPRAY_SEC, max(0.0, float(structured["spray_seconds"]))))

    return structured

# ---------- blue/green ND processing ----------
def blue_index_process(in_path: Path, mask_path: Path, overlay_path: Path):
    img_full = cv2.imread(str(in_path))
    if img_full is None:
        raise RuntimeError(f"Failed to read image {in_path}")
    h_full, w_full = img_full.shape[:2]
    proc_scale = PROCESS_SCALE
    if w_full <= MIN_PROC_WIDTH:
        proc_scale = 1.0
    proc_w = max(2, int(w_full * proc_scale))
    proc_h = max(2, int(h_full * proc_scale))
    img_proc = cv2.resize(img_full, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
    imgf = img_proc.astype(np.float32)
    B = imgf[:, :, 0]; G = imgf[:, :, 1]; R = imgf[:, :, 2]
    denom = (B + 0.5 * (R + G)) + 1e-6
    nd = (B - 0.5 * (R + G)) / denom
    nd_scaled = ((nd + 1.0) / 2.0) * 255.0
    nd_u8 = np.clip(nd_scaled, 0, 255).astype(np.uint8)
    _, mask_small = cv2.threshold(nd_u8, BLUE_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask_full = cv2.resize(mask_small, (w_full, h_full), interpolation=cv2.INTER_NEAREST)
    percent_flagged = (mask_full > 0).sum() / mask_full.size * 100.0
    cv2.imwrite(str(mask_path), mask_full)
    blue_color = np.zeros_like(img_full)
    blue_color[:, :, 0] = mask_full
    overlay = cv2.addWeighted(img_full, 1.0, blue_color, 0.6, 0)
    try:
        contours, _ = cv2.findContours(mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            scale_x = w_full / proc_w
            scale_y = h_full / proc_h
            scaled_contours = []
            for c in contours:
                c = c.astype(np.float32)
                c[:, 0, 0] = c[:, 0, 0] * scale_x
                c[:, 0, 1] = c[:, 0, 1] * scale_y
                c = c.astype(np.int32)
                scaled_contours.append(c)
            cv2.drawContours(overlay, scaled_contours, -1, (255, 255, 255), 1)
    except Exception:
        pass
    cv2.imwrite(str(overlay_path), overlay)
    return percent_flagged, mask_full, overlay

# ---------- rclone ----------
def rclone_copyto(local: Path, remote_folder: str, remote_name: str, dry_run: bool = False) -> bool:
    rclone_path = shutil_which("rclone")
    if not rclone_path:
        log("rclone binary not found; skipping upload.")
        return False
    if not local.exists():
        log(f"rclone: local file missing, skipping upload: {local}")
        return False
    user_conf = Path("/home/sih/.config/rclone/rclone.conf")
    target = f"{remote_folder.rstrip('/')}/{remote_name}"
    cmd = [rclone_path]
    if user_conf.exists():
        cmd += ["--config", str(user_conf)]
    cmd += ["copyto", str(local), target, "--no-traverse"]
    if dry_run:
        cmd += ["--dry-run"]
    log(f"rclone cmd: {' '.join(cmd)}")
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        stdout = p.stdout.decode().strip()
        stderr = p.stderr.decode().strip()
        if stdout:
            log(f"rclone stdout: {stdout}")
        if p.returncode != 0:
            log(f"rclone failed (code {p.returncode}): {stderr if stderr else '(no stderr)'}")
            return False
        log("rclone upload succeeded.")
        return True
    except subprocess.TimeoutExpired:
        log("rclone timed out.")
        return False
    except Exception as e:
        log(f"rclone exception: {e}")
        return False

# ---------- servo control (Option 2) ----------
def servo_init_with_pigs():
    if not PIGS_AVAILABLE:
        return None
    def pigs_set(position):
        pos = float(position)
        pos = max(-1.0, min(1.0, pos))
        # map to 1000..2000us (1500 center)
        pulse = int(1500 + pos * 500)
        # use pigs sr (servo) which expects microseconds; fallback to pigs pwm is possible
        subprocess.run(["pigs", "sr", str(SERVO_GPIO), str(pulse)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return pulse
    def pigs_detach():
        subprocess.run(["pigs", "sr", str(SERVO_GPIO), "0"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {"set": pigs_set, "detach": pigs_detach, "type": "pigs"}

def servo_init_with_gpiozero():
    if not GPIOZERO_AVAILABLE or Servo is None:
        return None
    try:
        if PiGPIOFactory is not None:
            factory = PiGPIOFactory()
            s = Servo(SERVO_GPIO, pin_factory=factory, min_pulse_width=0.6/1000, max_pulse_width=2.4/1000)
            return {"obj": s, "type": "gpiozero", "factory": "pigpio"}
        else:
            s = Servo(SERVO_GPIO, min_pulse_width=0.6/1000, max_pulse_width=2.4/1000)
            return {"obj": s, "type": "gpiozero", "factory": "native"}
    except Exception as e:
        log(f"gpiozero Servo init failed: {e}")
        return None

def init_servo_controller():
    pigs_ctl = servo_init_with_pigs()
    if pigs_ctl:
        log(f"Servo will be controlled via pigs on GPIO {SERVO_GPIO}")
        return ("pigs", pigs_ctl)
    gz = servo_init_with_gpiozero()
    if gz:
        log(f"Servo initialized via gpiozero (factory={gz.get('factory')}) on GPIO {SERVO_GPIO}")
        return ("gpiozero", gz)
    log("No servo controller available")
    return (None, None)

def move_servo(controller_tuple, position):
    typ, ctl = controller_tuple
    try:
        if typ == "pigs":
            pulse = ctl["set"](position)
            log(f"Servo moved to {position} (pulse {pulse}us) via pigs")
            time.sleep(SERVO_SETTLE_SEC)
            try:
                ctl["detach"]()
            except Exception:
                pass
            return True
        elif typ == "gpiozero":
            s = ctl["obj"]
            s.value = position
            log(f"Servo moved to {position} via gpiozero")
            time.sleep(SERVO_SETTLE_SEC)
            try:
                s.detach()
            except Exception:
                pass
            return True
        else:
            log("Servo controller missing; cannot move")
            return False
    except Exception as e:
        log(f"Servo move error: {e}")
        return False

# ---------- Pump controller (Pi drives pump directly) ----------
class PumpController:
    def __init__(self, pin=None):
        # pin: BCM pin number for pump control
        self.pin = int(pin) if pin is not None else None
        self.simulated = True
        self.pump = None
        # prefer gpiozero OutputDevice for safe switching
        if GPIOZERO_AVAILABLE and OutputDevice is not None and self.pin is not None:
            try:
                self.pump = OutputDevice(self.pin)
                # default to simulated True for safety; only enable when --do-spray used
                self.simulated = True
                log(f"Pump OutputDevice initialized on GPIO {self.pin} (simulated until enabled).")
            except Exception as e:
                log(f"Pump init failed (GPIO) - falling back to simulation: {e}")
                self.pump = None
                self.simulated = True
        else:
            log("GPIO or OutputDevice not available - pump will be simulated")
            self.pump = None
            self.simulated = True

    def enable_real(self):
        if self.pump is not None:
            self.simulated = False
            log("PumpController: spraying ENABLED (real mode)")
        else:
            log("PumpController: real pump not available; staying simulated")

    def spray(self, seconds):
        sec = float(seconds)
        if sec <= 0:
            log("spray called with non-positive duration; skipping")
            return {"sprayed": False, "reason": "non_positive_duration"}
        sec = min(MAX_SPRAY_SEC, sec)
        log(f"Spray requested: {sec:.2f}s (simulated={self.simulated})")
        if self.simulated:
            log(f"[SIMULATED PUMP] ON for {sec:.2f}s")
            time.sleep(min(sec, 0.5))
            log("[SIMULATED PUMP] OFF")
            return {"sprayed": True, "simulated": True, "duration": sec}
        try:
            self.pump.on()
            time.sleep(sec)
            self.pump.off()
            log("Pump spray complete")
            return {"sprayed": True, "simulated": False, "duration": sec}
        except Exception as e:
            log(f"Pump operation failed: {e}")
            return {"sprayed": False, "error": str(e)}

# ---------- helpers ----------
def ingest_uploaded_image(src: str, dest_base: Path, pos_name: str, ts: str) -> Path:
    src_path = Path(src).expanduser()
    if not src_path.exists():
        raise FileNotFoundError(f"Uploaded file not found: {src}")
    dest = dest_base / f"{pos_name}_{ts}.jpg"
    shutil_copy(str(src_path), str(dest))
    return dest

def crop_largest_mask_region(image_full, mask_full, pad_percent=0.10):
    try:
        contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            h, w = image_full.shape[:2]
            side = min(h, w)
            cx, cy = w // 2, h // 2
            half = side // 4
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(w, cx + half)
            y2 = min(h, cy + half)
            return image_full[y1:y2, x1:x2]
        c = max(contours, key=lambda x: cv2.contourArea(x))
        x, y, w, h = cv2.boundingRect(c)
        pad_x = int(w * pad_percent)
        pad_y = int(h * pad_percent)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(image_full.shape[1], x + w + pad_x)
        y2 = min(image_full.shape[0], y + h + pad_y)
        return image_full[y1:y2, x1:x2]
    except Exception:
        h, w = image_full.shape[:2]
        side = min(h, w)
        cx, cy = w // 2, h // 2
        half = side // 4
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)
        return image_full[y1:y2, x1:x2]

# ---------- MAIN ----------
def main():
    ensure_dirs_and_counters()
    log("=== SIH end-to-end startup ===")

    # args
    do_spray = "--do-spray" in sys.argv
    dry_run_upload = "--dry-run-upload" in sys.argv
    mode_arg = None
    if "--mode" in sys.argv:
        try:
            i = sys.argv.index("--mode"); mode_arg = sys.argv[i+1]
        except Exception:
            mode_arg = "camera"
    upload_paths_arg = None
    if "--upload-paths" in sys.argv:
        try:
            i = sys.argv.index("--upload-paths"); upload_paths_arg = sys.argv[i+1]
        except Exception:
            upload_paths_arg = None

    # load model
    interpreter = load_tflite_model(MODEL_PATH)

    # pump controller
    pump = PumpController(pin=PUMP_GPIO)
    if do_spray:
        pump.enable_real()

    # mode selection
    mode = "camera"
    if mode_arg:
        if mode_arg.lower() in ("camera","capture","c"):
            mode = "camera"
        elif mode_arg.lower() in ("upload","u"):
            mode = "upload"
    else:
        try:
            choice = input("Select input method - type 'camera' to capture or 'upload' to supply images: ").strip().lower()
            if choice in ("upload","u"):
                mode = "upload"
            else:
                mode = "camera"
        except Exception:
            mode = "camera"

    upload_paths = {"left": None, "center": None, "right": None}
    if mode == "upload":
        if upload_paths_arg:
            parts = [p.strip() for p in upload_paths_arg.split(",") if p.strip()]
            if len(parts) == 1:
                upload_paths = {k: parts[0] for k in upload_paths}
            elif len(parts) >= 3:
                upload_paths["left"], upload_paths["center"], upload_paths["right"] = parts[:3]
            elif len(parts) == 2:
                upload_paths["left"] = parts[0]; upload_paths["center"] = parts[1]; upload_paths["right"] = parts[1]
        else:
            try:
                reply = input("Enter image paths (single or 3 comma-separated): ").strip()
                if reply:
                    parts = [p.strip() for p in reply.split(",") if p.strip()]
                    if len(parts) == 1:
                        upload_paths = {k: parts[0] for k in upload_paths}
                    elif len(parts) >= 3:
                        upload_paths["left"], upload_paths["center"], upload_paths["right"] = parts[:3]
                    elif len(parts) == 2:
                        upload_paths["left"] = parts[0]; upload_paths["center"] = parts[1]; upload_paths["right"] = parts[1]
                else:
                    log("No upload paths provided; switching to camera mode")
                    mode = "camera"
            except Exception:
                mode = "camera"

    # servo init (Option 2)
    servo_ctrl = init_servo_controller()

    positions = [("left", -1), ("center", 0), ("right", 1)]
    results = {}

    for name, posval in positions:
        log(f"--- Move -> {name} ---")
        servo_moved = False
        if servo_ctrl[0] is not None:
            servo_moved = move_servo(servo_ctrl, posval)
        else:
            log("Servo unavailable or not initialized")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_orig = IMG_DIR / f"{name}_{ts}.jpg"

        if mode == "upload":
            try:
                src = upload_paths.get(name)
                if not src:
                    log(f"No upload image provided for {name}")
                    results[name] = {"status": "capture_failed", "reason": "no_upload_path", "servo_moved": servo_moved}
                    continue
                local_orig = ingest_uploaded_image(src, IMG_DIR, name, ts)
                log(f"Uploaded image ingested for {name}: {local_orig}")
            except Exception as e:
                log(f"Ingest/upload error for {name}: {e}")
                results[name] = {"status": "capture_failed", "reason": "ingest_failed", "servo_moved": servo_moved}
                continue
        else:
            if not take_photo(local_orig):
                log("Capture failed")
                results[name] = {"status": "capture_failed", "servo_moved": servo_moved}
                continue
            log(f"Captured {local_orig.name}")

        mask_path = IMG_DIR / f"{name}_{ts}_mask.jpg"
        overlay_path = IMG_DIR / f"{name}_{ts}_overlay.jpg"
        crop_path = IMG_DIR / f"{name}_{ts}_crop.jpg"
        ml_json_path = IMG_DIR / f"{name}_{ts}_ml.json"

        try:
            stress_percent, mask_full, overlay = blue_index_process(local_orig, mask_path, overlay_path)
            log(f"Processed: stress={stress_percent:.2f}%")
        except Exception as e:
            log("Processing error: " + str(e))
            results[name] = {"status": "processing_failed", "servo_moved": servo_moved}
            continue

        ml_result = None
        spray_result = None

        if stress_percent >= STRESS_THRESHOLD_PERCENT and interpreter is not None:
            try:
                img_full = cv2.imread(str(local_orig))
                crop = crop_largest_mask_region(img_full, mask_full, pad_percent=0.12)
                try:
                    cv2.imwrite(str(crop_path), crop)
                    log(f"Saved inference crop: {crop_path}")
                except Exception as e:
                    log(f"Failed to save crop image: {e}")
                ml_result = run_tflite_on_crop(interpreter, crop)
                try:
                    with open(ml_json_path, "w") as f:
                        json.dump(ml_result, f)
                    log(f"Saved ML JSON: {ml_json_path}")
                except Exception as e:
                    log(f"Failed to save ML JSON: {e}")
                log(f"ML inference structured: {ml_result.get('structured') if isinstance(ml_result, dict) else ml_result}")
            except Exception as e:
                log(f"ML processing error for {name}: {e}")
                ml_result = {"error": str(e)}
        elif stress_percent >= STRESS_THRESHOLD_PERCENT and interpreter is None:
            log("Interpreter missing; skipping ML inference.")
            ml_result = {"error": "interpreter_missing_or_model_not_loaded"}

        if isinstance(ml_result, dict) and "structured" in ml_result:
            structured = ml_result["structured"]
            spray_decision = structured.get("spray_decision", False)
            spray_seconds = structured.get("spray_seconds", None)
            if structured.get("spray_probability") is not None:
                if structured["spray_probability"] < SPRAY_PROBABILITY_MIN:
                    spray_decision = False
            if spray_decision:
                if spray_seconds is None:
                    if structured.get("severity") is not None and structured["severity"] >= 0.6:
                        spray_seconds = DEFAULT_SPRAY_HIGH_SEC
                    else:
                        spray_seconds = DEFAULT_SPRAY_LOW_SEC
                spray_seconds = float(min(MAX_SPRAY_SEC, max(0.0, spray_seconds)))
                spray_result = pump.spray(spray_seconds)
            else:
                log("Model decided no spray required.")
                spray_result = {"sprayed": False, "reason": "model_decision_no_spray"}
        else:
            if stress_percent >= STRESS_THRESHOLD_PERCENT:
                log("Stress high but no valid ML output; skipping spraying (conservative).")
                spray_result = {"sprayed": False, "reason": "no_ml_output"}
            else:
                spray_result = {"sprayed": False, "reason": "stress_below_threshold"}

        prefix = pos_to_prefix[name]
        idx = get_and_inc(prefix)
        orig_name = f"{prefix}{idx}.jpg"
        overlay_name = f"{prefix}{idx}_overlay.jpg"
        mask_name = f"{prefix}{idx}_mask.jpg"
        crop_name = f"{prefix}{idx}_crop.jpg"
        ml_name = f"{prefix}{idx}_ml.json"

        upl1 = rclone_copyto(local_orig, RCLONE_ORIG, orig_name, dry_run=dry_run_upload)
        upl2 = rclone_copyto(overlay_path, RCLONE_PROC, overlay_name, dry_run=dry_run_upload)
        upl3 = rclone_copyto(mask_path, RCLONE_PROC, mask_name, dry_run=dry_run_upload)
        upl4 = False
        upl5 = False
        if crop_path.exists():
            upl4 = rclone_copyto(crop_path, RCLONE_PROC, crop_name, dry_run=dry_run_upload)
        if ml_json_path.exists():
            upl5 = rclone_copyto(ml_json_path, RCLONE_PROC, ml_name, dry_run=dry_run_upload)

        results[name] = {
            "status": "ok",
            "servo_moved": servo_moved,
            "stress_percent": stress_percent,
            "ml_result": ml_result,
            "spray_result": spray_result,
            "remote_orig": orig_name if upl1 else None,
            "remote_overlay": overlay_name if upl2 else None,
            "remote_mask": mask_name if upl3 else None,
            "remote_crop": crop_name if upl4 else None,
            "remote_ml_json": ml_name if upl5 else None
        }

        # cleanup local files if uploaded successfully
        for p in (local_orig, overlay_path, mask_path):
            try:
                if p.exists() and (p != crop_path or upl4):
                    p.unlink()
            except Exception:
                pass
        try:
            if crop_path.exists() and upl4:
                crop_path.unlink()
        except Exception:
            pass
        try:
            if ml_json_path.exists() and upl5:
                ml_json_path.unlink()
        except Exception:
            pass

        time.sleep(0.3)

    log("=== SCAN CYCLE COMPLETE ===")
    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    main()
