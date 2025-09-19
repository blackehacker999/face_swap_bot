#!/usr/bin/env python3
# face_swap_bot.py
# Telegram face-swap bot (uses Haar cascades + seamlessClone)
# Credit banner & credit line as requested.

import os
import io
import time
import tempfile
import requests
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import telebot
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

# =========================
# Banner (Green Color)
# =========================
banner = r"""
 █████╗ ███╗   ███╗ █████╗ ██████╗      ██╗██╗████████╗
██╔══██╗████╗ ████║██╔══██╗██╔══██╗     ██║██║╚══██╔══╝
███████║██╔████╔██║███████║██████╔╝     ██║██║   ██║   
██╔══██║██║╚██╔╝██║██╔══██║██╔══██╗██   ██║██║   ██║   
██║  ██║██║ ╚═╝ ██║██║  ██║██║  ██║╚█████╔╝██║   ██║   
╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚════╝ ╚═╝   ╚═╝     
"""
print(Fore.GREEN + banner)
print(f"{Fore.RED}{Style.BRIGHT}➪ Credit by CYBER AMARJIT\n")

# =========================
# Read token function as requested
# =========================
def read_token(path='token.txt'):
    if not os.path.exists(path):
        print(f"[ERROR] Token file '{path}' not found. Create it and put your BOT token inside.")
        raise SystemExit(1)
    with open(path, 'r') as f:
        token = f.read().strip()
    if not token:
        print("[ERROR] Token file is empty.")
        raise SystemExit(1)
    return token

BOT_TOKEN = read_token()
bot = telebot.TeleBot(BOT_TOKEN, parse_mode=None)

# =========================
# Per-chat temporary storage
# =========================
# Stores {'face_path': str}
CHAT_STORE = {}
TMP_DIR = Path(tempfile.gettempdir()) / "tg_face_swap_bot"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Haar cascade (OpenCV data)
# =========================
haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
if not os.path.exists(haar_path):
    print("[WARN] Haar cascade not found in cv2.data.haarcascades. Face detection might fail.")
face_cascade = cv2.CascadeClassifier(haar_path)

# =========================
# Helper functions
# =========================
def download_photo(file_info):
    file_path = bot.get_file(file_info.file_id).file_path
    file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
    r = requests.get(file_url, stream=True)
    if r.status_code != 200:
        raise RuntimeError("Failed to download file from Telegram")
    return io.BytesIO(r.content)

def detect_largest_face(img_gray):
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
    if len(faces) == 0:
        return None
    # choose largest area
    faces = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)
    return faces[0]  # x,y,w,h

def extract_face_and_mask(src_bgr, face_box, padding=0.3):
    x, y, w, h = face_box
    # expand box slightly
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(src_bgr.shape[1], x + w + pad_x)
    y2 = min(src_bgr.shape[0], y + h + pad_y)
    face_roi = src_bgr[y1:y2, x1:x2].copy()
    mask = np.zeros(face_roi.shape[:2], dtype=np.uint8)
    # create an elliptical mask to approximate face shape
    center = (face_roi.shape[1]//2, face_roi.shape[0]//2)
    axes = (int(face_roi.shape[1]*0.45), int(face_roi.shape[0]*0.55))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return face_roi, mask, (x1, y1, x2, y2)

def seamless_paste(face_img, face_mask, target_img, target_face_box):
    tx, ty, tw, th = target_face_box
    # center where to place
    center = (tx + tw//2, ty + th//2)
    # resize face_img to target face bounding size
    # compute scale factor to fit width
    scale_w = tw / face_img.shape[1]
    scale_h = th / face_img.shape[0]
    scale = min(scale_w, scale_h) * 1.05  # slight oversize for coverage
    new_w = max(1, int(face_img.shape[1] * scale))
    new_h = max(1, int(face_img.shape[0] * scale))
    face_resized = cv2.resize(face_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(face_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # create a 3-channel mask
    mask3 = cv2.merge([mask_resized, mask_resized, mask_resized])
    # compute a ROI center for clone (already center in target coords)
    output = cv2.seamlessClone(face_resized, target_img, mask_resized, center, cv2.NORMAL_CLONE)
    return output

# =========================
# Bot Commands & Handlers
# =========================
@bot.message_handler(commands=['start'])
def cmd_start(message):
    bot.send_message(message.chat.id,
                     "Salam! Mujhe pehle photo (source face) bhejein — phir doosra photo bhejein jisme aap face swap karwana chahte hain.\n\n"
                     "Commands:\n/reset - reset stored source photo\n/help - instructions")

@bot.message_handler(commands=['help'])
def cmd_help(message):
    bot.send_message(message.chat.id,
                     "Usage:\n1) Send a photo that contains the face you want to transfer (source).\n2) Send the second photo where you want that face to be placed (target).\nIf face detection fails, try clearer, frontal faces.\nUse /reset to clear stored source photo.")

@bot.message_handler(commands=['reset'])
def cmd_reset(message):
    cid = message.chat.id
    if cid in CHAT_STORE:
        try:
            p = CHAT_STORE[cid].get('face_path')
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
        CHAT_STORE.pop(cid, None)
    bot.send_message(cid, "Reset complete. Ab aap phir se source photo bhej sakte hain.")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    cid = message.chat.id
    # get highest-resolution photo
    file_info = message.photo[-1]
    try:
        bio = download_photo(file_info)
    except Exception as e:
        bot.send_message(cid, "Failed to download photo. Try again.")
        return

    # save temporarily
    timestamp = int(time.time()*1000)
    fname = TMP_DIR / f"{cid}_{timestamp}.jpg"
    with open(fname, 'wb') as f:
        f.write(bio.getbuffer())

    # If we don't yet have a source face for this chat, treat this as source
    if cid not in CHAT_STORE or 'face_path' not in CHAT_STORE[cid]:
        # attempt to detect face in this image
        img_bgr = cv2.imdecode(np.fromfile(str(fname), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            bot.send_message(cid, "Could not read the image. Please try again.")
            return
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        face_box = detect_largest_face(gray)
        if face_box is None:
            bot.send_message(cid, "Face not detected in this photo. Please send a clearer frontal face as the source.")
            # cleanup file
            try: os.remove(fname)
            except: pass
            return
        # store as source
        CHAT_STORE[cid] = {'face_path': str(fname)}
        bot.send_message(cid, "✅ Source face saved. Ab dusra (target) photo bhejein jisme aap face swap karwana chahte hain.")
        return
    else:
        # We have source; treat this as target and perform swap
        source_path = CHAT_STORE[cid].get('face_path')
        if not source_path or not os.path.exists(source_path):
            # fallback — ask to re-send source
            CHAT_STORE.pop(cid, None)
            bot.send_message(cid, "Source photo missing. Please send the source face photo again.")
            # cleanup this file
            try: os.remove(fname)
            except: pass
            return

        # load images
        src_bgr = cv2.imdecode(np.fromfile(source_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        tgt_bgr = cv2.imdecode(np.fromfile(str(fname), dtype=np.uint8), cv2.IMREAD_COLOR)
        if src_bgr is None or tgt_bgr is None:
            bot.send_message(cid, "Error reading images. Try again.")
            return

        # detect faces
        src_gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
        tgt_gray = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2GRAY)
        src_box = detect_largest_face(src_gray)
        tgt_box = detect_largest_face(tgt_gray)

        if src_box is None:
            bot.send_message(cid, "Could not detect face in the stored source image. Please send a clearer source photo.")
            CHAT_STORE.pop(cid, None)
            return
        if tgt_box is None:
            bot.send_message(cid, "Could not detect face in the target image. Please send a different target photo with a clear frontal face.")
            return

        # extract face and mask from source
        face_roi, face_mask, coords = extract_face_and_mask(src_bgr, src_box)
        try:
            result = seamless_paste(face_roi, face_mask, tgt_bgr, tgt_box)
        except Exception as e:
            bot.send_message(cid, f"Error during blending: {e}")
            return

        # save result to temp and send
        out_path = TMP_DIR / f"{cid}_result_{timestamp}.jpg"
        cv2.imencode('.jpg', result)[1].tofile(str(out_path))

        with open(out_path, 'rb') as f:
            bot.send_photo(cid, f, caption="Here's your face-swapped image ✅\nIf you want to do another swap, send a new source photo (it will overwrite the stored one). Use /reset to clear.")
        # cleanup stored source to require a new source next time (optional)
        try:
            os.remove(source_path)
        except Exception:
            pass
        CHAT_STORE.pop(cid, None)
        # cleanup intermediate files
        try:
            os.remove(str(fname))
        except Exception:
            pass
        try:
            os.remove(str(out_path))
        except Exception:
            pass

# fallback for other messages
@bot.message_handler(func=lambda message: True)
def fallback(message):
    bot.send_message(message.chat.id, "Send a photo to start. Use /help for instructions.")

# =========================
# Run polling
# =========================
if __name__ == '__main__':
    print("[INFO] Bot started. Listening for messages...")
    try:
        bot.infinity_polling(timeout=60, long_polling_timeout = 90)
    except KeyboardInterrupt:
        print("\n[INFO] Exiting (KeyboardInterrupt).")
    except Exception as ex:
        print(f"[ERROR] Bot crashed: {ex}")
