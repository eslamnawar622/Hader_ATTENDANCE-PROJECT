#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import pickle
import json
import csv
import hashlib
import random
import subprocess
import threading
import time
from datetime import datetime, date, timedelta

import cv2
import numpy as np
try:
    import pygame
    pygame.mixer.init()
except Exception:
    pygame = None

import customtkinter as ctk
from PIL import Image

# ==========================================
#  1) كشف GPU تلقائي (CNN لو موجود، HOG لو لأ)
# ==========================================
def detect_gpu():
    try:
        import dlib
        if hasattr(dlib, 'cuda') and dlib.cuda.get_num_devices() > 0:
            print("[GPU CHECK] CUDA found, using CNN model.")
            return True
    except Exception:
        pass
    print("[GPU CHECK] No CUDA / GPU. Falling back to CPU (HOG).")
    return False

USE_GPU = detect_gpu()
FACE_LOCATION_MODEL = "cnn" if USE_GPU else "hog"

# ==========================================
#  2) المسارات (تشتغل من سورس أو من exe)
# ==========================================
if getattr(sys, 'frozen', False):
    BASE_DIR = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR   = os.path.join(BASE_DIR, "data")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
LOGS_DIR   = os.path.join(BASE_DIR, "logs")
CACHE_DIR  = os.path.join(BASE_DIR, "tts_cache")
EMP_DIR    = os.path.join(BASE_DIR, "employee")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EMP_DIR, exist_ok=True)

WEIGHTS_FILE    = os.path.join(DATA_DIR, "axis_face_weights.pkl")
SETTINGS_FILE   = os.path.join(CONFIG_DIR, "axis_settings.json")
FINES_FILE      = os.path.join(DATA_DIR, "employee_fines.json")

# لو في ملفات قديمة بره الفولدر
_OLD_W = os.path.join(BASE_DIR, "..", "axis_face_weights.pkl")
_OLD_S = os.path.join(BASE_DIR, "..", "axis_settings.json")
if os.path.exists(_OLD_W) and not os.path.exists(WEIGHTS_FILE):
    shutil.copy2(_OLD_W, WEIGHTS_FILE)
if os.path.exists(_OLD_S) and not os.path.exists(SETTINGS_FILE):
    shutil.copy2(_OLD_S, SETTINGS_FILE)

import face_recognition

TOLERANCE       = 0.45
PROCESS_EVERY_N = 2
FRAME_SCALE     = 0.25
FINE_PER_MINUTE = 10  # كل دقيقة تأخير = 10 جنيه

COLORS = {
    "bg": "#0f0f1a", "card": "#1a1a2e", "accent": "#16213e",
    "primary": "#e94560", "success": "#00ff88", "warning": "#ffaa00",
    "danger": "#ff3333", "info": "#00d4ff", "text": "#ffffff",
    "muted": "#888888"
}

# ==========================================
#           التطبيق الرئيسي
# ==========================================
class AxisProAttendance:
    def __init__(self, root):
        self.root = root
        self.root.title("⚡ Axis Design Studio | Smart Attendance")
        self.root.geometry("600x500")
        self.root.configure(fg_color=COLORS["bg"])
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.is_running    = False
        self.latest_frame  = None
        self.frame_lock    = threading.Lock()
        self.fps           = 0
        self.frame_count   = 0
        self.fps_time      = time.time()
        self.last_greeted  = {}
        self.latest_results= None

        self.known_encodings = []
        self.known_names     = []
        self.known_roles     = []
        self.known_audio     = []
        self.employee_dir    = self.find_employee_dir()
        self.on_time_limit   = self.load_settings()

        status = self.check_and_build_weights()
        if status == "loaded":
            self.show_main_menu()
        elif status == "error":
            pass

    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r") as f:
                    return json.load(f).get("on_time", "09:00")
            except:
                pass
        return "09:00"

    def save_settings(self, val):
        self.on_time_limit = val
        with open(SETTINGS_FILE, "w") as f:
            json.dump({"on_time": val}, f)

    def format_12h(self, time_24h):
        h, m = map(int, time_24h.split(':'))
        ampm = "AM" if h < 12 else "PM"
        h12 = h % 12
        if h12 == 0:
            h12 = 12
        return f"{h12:02d}:{m:02d} {ampm}"

    def find_employee_dir(self):
        if os.path.isdir(EMP_DIR):
            print(f"[INFO] employee dir: {EMP_DIR}")
            return EMP_DIR
        alt = os.path.join(BASE_DIR, "..", "employee")
        if os.path.isdir(alt):
            print(f"[INFO] employee dir (alt): {alt}")
            return alt
        return None

    def scan_employees(self):
        data = {}
        if not self.employee_dir:
            return data
        for role_entry in os.listdir(self.employee_dir):
            role_path = os.path.join(self.employee_dir, role_entry)
            if not os.path.isdir(role_path):
                continue
            role_name = role_entry.replace("_", " ").replace("-", " ")
            for emp_entry in os.listdir(role_path):
                emp_path = os.path.join(role_path, emp_entry)
                if not os.path.isdir(emp_path):
                    continue
                images, audio = [], None
                for f in os.listdir(emp_path):
                    low = f.lower()
                    if low.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                        images.append(os.path.join(emp_path, f))
                    elif audio is None and low.endswith((".mp3", ".m4a", ".wav", ".ogg", ".flac", ".aac")):
                        audio = os.path.join(emp_path, f)
                if images:
                    data[emp_entry] = {
                        "name": emp_entry,
                        "images": images,
                        "audio": audio,
                        "role": role_name,
                        "path": emp_path
                    }
        return data

    def check_and_build_weights(self):
        current = self.scan_employees()
        if os.path.exists(WEIGHTS_FILE) and len(current) > 0:
            try:
                with open(WEIGHTS_FILE, "rb") as f:
                    saved = pickle.load(f)
                saved_names = set(saved.get("names", []))
                cur_names   = set(current.keys())
                if saved_names == cur_names:
                    self.known_encodings = saved["encodings"]
                    self.known_names     = saved["names"]
                    self.known_roles     = saved["roles"]
                    self.known_audio     = saved["audio_paths"]
                    print(f"[INFO] Loaded weights for {len(self.known_names)} employees.")
                    return "loaded"
            except Exception as e:
                print("[WARN] Corrupt weights, retraining...", e)
        if not current:
            msg = (f"❌ No employee folders found!\n\n"
                   f"Searched: {self.employee_dir or 'N/A'}\n\n"
                   f"Expected:\n"
                   f"  employee/Role_Name/Employee_Name/photo.jpg")
            self.show_error(msg)
            return "error"
        self.train_model(current)
        return "training"

    def train_model(self, employees):
        self.clear_screen()
        self.root.geometry("600x400")
        ctk.CTkLabel(self.root, text="⚡ AXIS AI TRAINING",
                     font=("Arial", 28, "bold"), text_color=COLORS["primary"]).pack(pady=(50, 10))
        self.train_lbl = ctk.CTkLabel(self.root, text="Initializing...", font=("Consolas", 16))
        self.train_lbl.pack(pady=20)
        self.train_prog = ctk.CTkProgressBar(self.root, width=400, height=20)
        self.train_prog.set(0)
        self.train_prog.pack(pady=10)
        threading.Thread(target=self._training_worker, args=(employees,), daemon=True).start()

    def _training_worker(self, employees):
        encodings, names, roles, audio_paths = [], [], [], []
        items = list(employees.items())
        total = len(items)
        for idx, (name, info) in enumerate(items):
            self.root.after(0, lambda n=name, i=idx, t=total: (
                self.train_lbl.configure(text=f"Training: {n}  ({i+1}/{t})"),
                self.train_prog.set((i + 1) / t)
            ))
            person_encs = []
            for img_path in info["images"]:
                try:
                    img = face_recognition.load_image_file(img_path)
                    locs = face_recognition.face_locations(img, model=FACE_LOCATION_MODEL)
                    if not locs:
                        continue
                    enc = face_recognition.face_encodings(img, locs, num_jitters=1)
                    if enc:
                        person_encs.append(enc[0])
                except Exception as e:
                    print(f"[SKIP] {img_path}: {e}")
            if person_encs:
                avg_enc = np.mean(person_encs, axis=0)
                encodings.append(avg_enc)
                names.append(name)
                roles.append(info["role"])
                audio_paths.append(info["audio"])
        with open(WEIGHTS_FILE, "wb") as f:
            pickle.dump({
                "encodings": encodings, "names": names,
                "roles": roles, "audio_paths": audio_paths
            }, f)
        self.known_encodings = encodings
        self.known_names     = names
        self.known_roles     = roles
        self.known_audio     = audio_paths
        self.root.after(500, self.show_main_menu)

    def show_main_menu(self):
        self.clear_screen()
        self.root.geometry("600x500")
        ctk.CTkLabel(self.root, text="⚡", font=("Arial", 60), text_color=COLORS["primary"]).pack(pady=(40, 0))
        ctk.CTkLabel(self.root, text="AXIS DESIGN STUDIO",
                     font=("Arial", 28, "bold"), text_color=COLORS["text"]).pack(pady=5)

        gpu_status = "🟢 GPU Mode" if USE_GPU else "🔵 CPU Mode"
        ctk.CTkLabel(self.root, text=f"{gpu_status} | Face Recognition Ready",
                     font=("Arial", 12), text_color=COLORS["muted"]).pack(pady=(0, 20))

        ctk.CTkButton(self.root, text="▶  START ATTENDANCE", fg_color=COLORS["primary"],
                      hover_color="#ff6b81", font=("Arial", 18, "bold"),
                      height=55, width=280, corner_radius=15, command=self.open_camera_window).pack(pady=10)
        ctk.CTkButton(self.root, text="⚙  SETTINGS", fg_color=COLORS["accent"],
                      hover_color=COLORS["info"], font=("Arial", 16, "bold"),
                      height=45, width=280, corner_radius=12, command=self.open_settings).pack(pady=10)
        ctk.CTkButton(self.root, text="🚪  EXIT", fg_color=COLORS["danger"],
                      hover_color="#cc0000", font=("Arial", 16, "bold"),
                      height=45, width=280, corner_radius=12, command=self.quit_app).pack(pady=10)

        cnt = len(self.known_names)
        ctk.CTkLabel(self.root, text=f"👥 Employees Registered: {cnt}",
                     font=("Consolas", 12),
                     text_color=COLORS["success"] if cnt > 0 else COLORS["warning"]).pack(pady=20)

    def quit_app(self):
        self.is_running = False
        if pygame:
            pygame.mixer.quit()
        self.root.destroy()
        sys.exit(0)

    def open_settings(self):
        self.clear_screen()
        self.root.geometry("500x400")
        ctk.CTkLabel(self.root, text="⚙ Attendance Settings",
                     font=("Arial", 24, "bold"), text_color=COLORS["primary"]).pack(pady=(40, 20))
        time_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        time_frame.pack(pady=10)
        ctk.CTkLabel(time_frame, text="🕐 On-Time Limit:", font=("Arial", 14)).pack(pady=(0, 10))
        picker_frame = ctk.CTkFrame(time_frame, fg_color="transparent")
        picker_frame.pack()

        self.hour_var = ctk.StringVar(value="9")
        ctk.CTkOptionMenu(picker_frame, values=[str(i) for i in range(1, 13)], variable=self.hour_var,
                          width=80, font=("Consolas", 16)).pack(side="left", padx=5)
        ctk.CTkLabel(picker_frame, text=":", font=("Arial", 20, "bold")).pack(side="left")
        self.minute_var = ctk.StringVar(value="00")
        ctk.CTkOptionMenu(picker_frame, values=["00", "15", "30", "45"], variable=self.minute_var,
                          width=80, font=("Consolas", 16)).pack(side="left", padx=5)
        self.ampm_var = ctk.StringVar(value="AM")
        ctk.CTkOptionMenu(picker_frame, values=["AM", "PM"], variable=self.ampm_var,
                          width=80, font=("Arial", 14, "bold")).pack(side="left", padx=10)

        current_display = self.format_12h(self.on_time_limit)
        ctk.CTkLabel(time_frame, text=f"Current: {current_display}",
                     font=("Consolas", 12), text_color=COLORS["muted"]).pack(pady=10)

        ctk.CTkButton(self.root, text="💾  SAVE & BACK", height=45, width=200, corner_radius=12,
                      fg_color=COLORS["success"], hover_color="#00cc66",
                      font=("Arial", 16, "bold"), command=self.save_time_and_home).pack(pady=30)
        ctk.CTkButton(self.root, text="⬅  BACK", height=40, width=150, corner_radius=10,
                      fg_color=COLORS["muted"], hover_color="#666666",
                      font=("Arial", 14), command=self.show_main_menu).pack(pady=10)

    def save_time_and_home(self):
        try:
            h = int(self.hour_var.get())
            m = self.minute_var.get()
            ampm = self.ampm_var.get()
            if ampm == "PM" and h != 12:
                h += 12
            elif ampm == "AM" and h == 12:
                h = 0
            time_24h = f"{h:02d}:{m}"
            datetime.strptime(time_24h, "%H:%M")
            self.save_settings(time_24h)
            self.show_main_menu()
        except ValueError:
            pass

    def open_camera_window(self):
        self.clear_screen()
        self.root.geometry("1100x850")
        top = ctk.CTkFrame(self.root, height=50, fg_color=COLORS["card"], corner_radius=0)
        top.pack(fill="x")
        top.pack_propagate(False)
        self.fps_lbl = ctk.CTkLabel(top, text="FPS: --", font=("Consolas", 16, "bold"),
                                    text_color=COLORS["success"])
        self.fps_lbl.pack(side="left", padx=20)
        self.status_lbl = ctk.CTkLabel(top, text="🔍 Waiting for face...", font=("Arial", 16),
                                       text_color=COLORS["warning"])
        self.status_lbl.pack(side="left", padx=20)
        display_time = self.format_12h(self.on_time_limit)
        ctk.CTkLabel(top, text=f"⏰ On Time: {display_time}",
                     font=("Consolas", 13), text_color=COLORS["muted"]).pack(side="right", padx=20)
        cam_card = ctk.CTkFrame(self.root, fg_color="#000000", corner_radius=20)
        cam_card.pack(expand=True, fill="both", padx=20, pady=15)
        self.video_label = ctk.CTkLabel(cam_card, text="")
        self.video_label.pack(expand=True, fill="both", padx=8, pady=8)
        self.info_card = ctk.CTkFrame(self.root, height=100, fg_color=COLORS["card"], corner_radius=15)
        self.info_card.pack(fill="x", padx=20, pady=(0, 10))
        self.info_card.pack_propagate(False)
        self.info_name = ctk.CTkLabel(self.info_card, text="--", font=("Arial", 20, "bold"))
        self.info_name.pack(side="left", padx=20, pady=20)
        self.info_role = ctk.CTkLabel(self.info_card, text="--", font=("Arial", 14))
        self.info_role.pack(side="left", padx=10, pady=20)
        self.info_status = ctk.CTkLabel(self.info_card, text="", font=("Arial", 16, "bold"))
        self.info_status.pack(side="right", padx=20, pady=20)
        bot = ctk.CTkFrame(self.root, height=55, fg_color=COLORS["card"], corner_radius=0)
        bot.pack(fill="x", side="bottom")
        bot.pack_propagate(False)
        ctk.CTkButton(bot, text="⏹  BACK TO MENU", height=40, width=180, corner_radius=10,
                      fg_color=COLORS["danger"], hover_color="#cc0000",
                      font=("Arial", 14, "bold"), command=self.stop_and_home).pack(side="left", padx=20, pady=8)
        ctk.CTkButton(bot, text="🚪  EXIT", height=40, width=120, corner_radius=10,
                      fg_color="#444444", hover_color=COLORS["danger"],
                      font=("Arial", 14, "bold"), command=self.quit_app).pack(side="right", padx=20, pady=8)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.is_running = True
        self.frame_counter = 0
        self.latest_results = None
        self.last_greeted = {}

        threading.Thread(target=self.capture_loop, daemon=True).start()
        self.update_ui()

    def capture_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            self.frame_counter += 1
            if self.frame_counter % PROCESS_EVERY_N == 0 and len(self.known_encodings) > 0:
                self.process_frame(frame)
            with self.frame_lock:
                self.latest_frame = frame

    def process_frame(self, frame):
        small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        try:
            locations = face_recognition.face_locations(rgb, model=FACE_LOCATION_MODEL)
        except Exception:
            locations = face_recognition.face_locations(rgb, model="hog")

        if not locations:
            with self.frame_lock:
                self.latest_results = None
            self.root.after(0, lambda: self.status_lbl.configure(
                text="🔍 Waiting for face...", text_color=COLORS["warning"]))
            return

        encs = face_recognition.face_encodings(rgb, locations)
        names, roles, statuses, colors = [], [], [], []
        now_time = datetime.now().strftime("%H:%M")
        is_early = now_time <= self.on_time_limit

        for enc in encs:
            dists = face_recognition.face_distance(self.known_encodings, enc)
            best_idx = np.argmin(dists)
            if dists[best_idx] <= TOLERANCE:
                name = self.known_names[best_idx]
                role = self.known_roles[best_idx]
                audio = self.known_audio[best_idx]
                status = "EARLY ✅" if is_early else "LATE ⚠️"
                color = COLORS["success"] if is_early else COLORS["warning"]
                self.handle_greeting(name, role, is_early, audio)
                self.log_attendance(name, role, status)
            else:
                name, role, status, color = "Unknown", "Visitor", "", COLORS["danger"]
            names.append(name)
            roles.append(role)
            statuses.append(status)
            colors.append(color)

        with self.frame_lock:
            self.latest_results = (locations, names, roles, statuses, colors)

        if names[0] != "Unknown":
            s = statuses[0]
            c = COLORS["success"] if "EARLY" in s else COLORS["warning"]
            self.root.after(0, lambda n=names[0], s=s, c=c:
                            self.status_lbl.configure(text=f"👤 {n} | {s}", text_color=c))

    def handle_greeting(self, name, role, is_early, custom_audio):
        now = time.time()
        if name in self.last_greeted and now - self.last_greeted[name] < 60:
            return
        self.last_greeted[name] = now
        threading.Thread(target=self._greeting_worker,
                         args=(name, role, is_early, custom_audio), daemon=True).start()

    def _greeting_worker(self, name, role, is_early, custom_audio):
        try:
            if is_early and custom_audio and os.path.exists(custom_audio) and pygame:
                self._play_audio(custom_audio)
                return
            if is_early:
                msgs = [
                    f"يااااه يا {name}، جاي بدري! كده حلو أوي يا فنان!",
                    f"صباح الفل يا {name}! إنتا جاي النهاردة بروح رياضية!",
                    f"يا وحش يا {name}، جاي بدري! الكل هيتعلم منك!",
                    f"عاش يا {name}! بدري ومتزنط! ربنا يباركلك!"
                ]
            else:
                # حساب الغرامة للصوت
                late_min, fine = self.calculate_late_minutes_and_fine()
                msgs = [
                    f"يا حبيبي يا {name}، إيه الريحة الحلوة دي! إحنا مستنينك!",
                    f"أخيراً ظهرت يا {name}! كنا هنبلغ عنك سرقة!",
                    f"يا {name}، إنتا جاي ولا جاي تسأل علينا؟ الجاية بدري شوية!",
                    f"صباح الخير يا {name}، أهلاً أهلاً! المرة الجاية لازم تيجي بدري!"
                ]
                if fine > 0:
                    msgs.append(f"يا {name}، إنت تأخرت {late_min} دقيقة! عليك {fine} جنيه غرامة!")
                    msgs.append(f"يا {name} يا فنان! التأخير ده كلفك {fine} جنيه!")
            text = random.choice(msgs)
            self._speak_egyptian(text)
        except Exception as e:
            print(f"[AUDIO ERROR] {e}")

    def _play_audio(self, path):
        if not pygame:
            return
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"[PLAY ERROR] {e}")

    def _speak_egyptian(self, text):
        try:
            h = hashlib.md5(text.encode()).hexdigest()
            cache = os.path.join(CACHE_DIR, f"tts_{h}.mp3")
            if not os.path.exists(cache):
                try:
                    cmd = [
                        "edge-tts",
                        "--voice", "ar-EG-HaniNeural",
                        "--text", text,
                        "--write-media", cache,
                        "--rate", "+20%",
                        "--pitch", "+0Hz"
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode != 0:
                        raise Exception("edge-tts failed")
                except:
                    try:
                        from gtts import gTTS
                        gTTS(text=text, lang='ar', slow=False).save(cache)
                    except Exception:
                        return
            if pygame:
                self._play_audio(cache)
        except Exception as e:
            print(f"[TTS ERROR] {e}")

    def calculate_late_minutes_and_fine(self):
        """بحسب دقايق التأخير والغرامة حسب الوقت الحالي"""
        now = datetime.now()
        limit_h, limit_m = map(int, self.on_time_limit.split(':'))
        limit = now.replace(hour=limit_h, minute=limit_m, second=0, microsecond=0)
        if now <= limit:
            return 0, 0
        diff = now - limit
        minutes = int(diff.total_seconds() // 60)
        fine = minutes * FINE_PER_MINUTE
        return minutes, fine

    def update_cumulative_fine(self, name, daily_fine, late_minutes):
        """يحدث ملف الـ JSON التراكمي ويرجع إجمالي الغرامات"""
        data = {}
        if os.path.exists(FINES_FILE):
            try:
                with open(FINES_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass
        
        if name not in data:
            data[name] = {"total_fines": 0.0, "total_minutes": 0, "entries": []}
        
        if daily_fine > 0:
            data[name]["total_fines"] += daily_fine
            data[name]["total_minutes"] += late_minutes
            data[name]["entries"].append({
                "date": date.today().isoformat(),
                "minutes": late_minutes,
                "fine": daily_fine
            })
            try:
                with open(FINES_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[FINES WRITE ERROR] {e}")
        
        return data[name]["total_fines"]

    def log_attendance(self, name, role, status):
        today = date.today().isoformat()
        now   = datetime.now().strftime("%H:%M:%S")
        daily_file = os.path.join(LOGS_DIR, f"attendance_{today}.csv")

        # ----- تأكد إن الشخص متسجلش قبل كده النهاردة -----
        already_logged = False
        if os.path.exists(daily_file):
            try:
                with open(daily_file, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("Name") == name:
                            already_logged = True
                            break
            except Exception:
                pass

        if already_logged:
            print(f"[LOG SKIP] {name} already logged today.")
            return

        # ----- حساب الغرامة -----
        late_minutes, daily_fine = 0, 0
        if "LATE" in status:
            late_minutes, daily_fine = self.calculate_late_minutes_and_fine()

        # ----- اجمع الغرامات القديمة + الجديدة -----
        total_fine = self.update_cumulative_fine(name, daily_fine, late_minutes)

        # ----- لو أول مرة نسجل -----
        exists = os.path.exists(daily_file)
        with open(daily_file, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["Date", "Time", "Name", "Role", "Status", "Late Minutes", "Daily Fine", "Total Fine"])
            w.writerow([today, now, name, role, status, late_minutes, daily_fine, total_fine])
            print(f"[LOGGED] {name} | {role} | {status} | Late: {late_minutes}min | Fine: {daily_fine} EGP | Total: {total_fine} EGP")

    def update_ui(self):
        if not self.is_running:
            return
        self.frame_count += 1
        t = time.time()
        if t - self.fps_time >= 1.0:
            self.fps = self.frame_count / (t - self.fps_time)
            self.frame_count = 0
            self.fps_time = t
            self.fps_lbl.configure(text=f"FPS: {self.fps:.1f}")
        with self.frame_lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
            results = self.latest_results
        if frame is None:
            self.root.after(15, self.update_ui)
            return
        display = frame.copy()
        if results is not None:
            locs, names, roles, statuses, colors = results
            scale = int(1 / FRAME_SCALE)
            for (top, right, bottom, left), name, role, status, color in zip(locs, names, roles, statuses, colors):
                top *= scale; right *= scale; bottom *= scale; left *= scale
                bgr = self.hex_to_bgr(color)
                cv2.rectangle(display, (left, top), (right, bottom), bgr, 3)

                if status and "LATE" in status:
                    display_status = "LATE"
                    display_color = COLORS["danger"]
                elif status and "EARLY" in status:
                    display_status = "EARLY"
                    display_color = COLORS["success"]
                else:
                    display_status = status
                    display_color = color

                y_txt = top - 80 if top > 80 else bottom + 10
                overlay = display.copy()
                cv2.rectangle(overlay, (left, y_txt), (left + 230, y_txt + 75), bgr, -1)
                cv2.addWeighted(overlay, 0.8, display, 0.2, 0, display)

                cv2.putText(display, name, (left + 8, y_txt + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(display, role, (left + 8, y_txt + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)
                if display_status:
                    bgr_status = self.hex_to_bgr(display_color)
                    cv2.putText(display, display_status, (left + 8, y_txt + 68),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr_status, 2)

                if name != "Unknown":
                    self.info_name.configure(text=name)
                    self.info_role.configure(text=role)
                    if "LATE" in str(status):
                        self.info_status.configure(text="LATE ⚠️", text_color=COLORS["danger"])
                    else:
                        self.info_status.configure(text="EARLY ✅", text_color=COLORS["success"])
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        ctk_img = ctk.CTkImage(pil, size=(640, 480))
        self.video_label.configure(image=ctk_img)
        self.video_label._image = ctk_img
        self.root.after(15, self.update_ui)

    def hex_to_bgr(self, hx):
        hx = hx.lstrip("#")
        r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
        return (b, g, r)

    def stop_and_home(self):
        self.is_running = False
        if hasattr(self, "cap") and self.cap:
            self.cap.release()
        self.show_main_menu()

    def clear_screen(self):
        for w in self.root.winfo_children():
            w.destroy()

    def show_error(self, msg):
        self.clear_screen()
        self.root.geometry("600x400")
        ctk.CTkLabel(self.root, text="⚠ ERROR", font=("Arial", 24, "bold"),
                     text_color=COLORS["danger"]).pack(pady=40)
        ctk.CTkLabel(self.root, text=msg, font=("Arial", 14), wraplength=500).pack(pady=10)

        def retry():
            if os.path.exists(WEIGHTS_FILE):
                os.remove(WEIGHTS_FILE)
            self.__init__(self.root)

        ctk.CTkButton(self.root, text="🔄 Retry", font=("Arial", 14), command=retry).pack(pady=10)
        ctk.CTkButton(self.root, text="🚪 Exit", font=("Arial", 14), fg_color=COLORS["danger"],
                      command=self.quit_app).pack(pady=10)

# ==========================================
#                التشغيل
# ==========================================
if __name__ == "__main__":
    root = ctk.CTk()
    app = AxisProAttendance(root)
    root.mainloop()