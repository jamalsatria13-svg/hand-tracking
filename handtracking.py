import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # butuh 2 tangan: kanan = gambar, kiri = hapus
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
success, frame = cap.read()
if not success:
    print("Gagal mengakses kamera.")
    exit()
frame = cv2.flip(frame, 1)
h, w = frame.shape[:2]

TIP_IDS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP_IDS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
WRIST_ID = 0

# Setiap goresan disimpan sebagai data (bukan langsung dibakar ke piksel),
# supaya bisa digeser (pan) tanpa kehilangan kualitas / terpotong permanen.
strokes = []  # list of {"p1": (x,y), "p2": (x,y), "color": (b,g,r), "thickness": int}

draw_color = (255, 255, 255)  # putih
base_thickness = 3
erase_radius = 35  # jarak toleransi untuk menghapus goresan di dekat jari

Z_NEAR, Z_FAR = -0.15, 0.05  # perkiraan rentang kedalaman dari MediaPipe (relatif, bukan presisi)

prev_right_point = None   # titik telunjuk kanan sebelumnya (untuk sambung garis)
prev_left_wrist = None    # posisi pergelangan kiri sebelumnya (untuk drag)
prev_right_wrist = None   # posisi pergelangan kanan sebelumnya (untuk drag)

print("KANAN: telunjuk saja = gambar | KIRI: telunjuk+tengah = hapus | KEPALAN (tangan mana saja) = geser gambar")
print("'c' = bersihkan semua | 'q' = keluar")


def fingers_up(hand_landmarks):
    """tip.y < pip.y = jari lurus ke atas. Heuristik sederhana, bisa keliru jika tangan miring."""
    result = {}
    for name in TIP_IDS:
        tip_y = hand_landmarks.landmark[TIP_IDS[name]].y
        pip_y = hand_landmarks.landmark[PIP_IDS[name]].y
        result[name] = tip_y < pip_y
    return result


def depth_to_thickness(z):
    t = np.clip((z - Z_NEAR) / (Z_FAR - Z_NEAR), 0.0, 1.0)
    return int(np.interp(t, [0, 1], [base_thickness + 3, max(1, base_thickness - 1)]))


def point_to_segment_dist(p, a, b):
    """Jarak titik p ke segmen garis a-b, dipakai untuk cek apakah goresan cukup dekat untuk dihapus."""
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    proj_x, proj_y = ax + t * dx, ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Gagal mengakses kamera.")
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    right_active = False
    left_active = False
    drag_delta = None  # (dx, dy) kalau ada tangan yang mengepal & bergerak

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # "Left" atau "Right"
            up = fingers_up(hand_landmarks)

            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1)
            )

            wrist = hand_landmarks.landmark[WRIST_ID]
            wx, wy = int(wrist.x * w), int(wrist.y * h)

            is_point = up["index"] and not up["middle"] and not up["ring"] and not up["pinky"]
            is_two = up["index"] and up["middle"] and not up["ring"] and not up["pinky"]
            is_fist = not any(up.values())

            # --- Mode DRAG: kepalan tangan (berlaku untuk tangan kanan ATAU kiri) ---
            if is_fist:
                if label == "Right":
                    if prev_right_wrist is not None:
                        drag_delta = (wx - prev_right_wrist[0], wy - prev_right_wrist[1])
                    prev_right_wrist = (wx, wy)
                else:
                    if prev_left_wrist is not None:
                        drag_delta = (wx - prev_left_wrist[0], wy - prev_left_wrist[1])
                    prev_left_wrist = (wx, wy)
                # Saat mengepal, jangan gambar/hapus dengan tangan yang sama
                if label == "Right":
                    prev_right_point = None
            else:
                if label == "Right":
                    prev_right_wrist = None
                else:
                    prev_left_wrist = None

            # --- Mode GAMBAR: hanya tangan KANAN, hanya telunjuk yang terangkat ---
            if label == "Right" and is_point and not is_fist:
                right_active = True
                index_tip = hand_landmarks.landmark[TIP_IDS["index"]]
                ix, iy, iz = int(index_tip.x * w), int(index_tip.y * h), index_tip.z
                thickness = depth_to_thickness(iz)

                cv2.circle(frame, (ix, iy), thickness + 2, draw_color, cv2.FILLED)

                if prev_right_point is not None:
                    strokes.append({
                        "p1": prev_right_point,
                        "p2": (ix, iy),
                        "color": draw_color,
                        "thickness": thickness
                    })
                prev_right_point = (ix, iy)
            elif label == "Right":
                prev_right_point = None

            # --- Mode HAPUS: hanya tangan KIRI, telunjuk + tengah terangkat ---
            if label == "Left" and is_two:
                left_active = True
                index_tip = hand_landmarks.landmark[TIP_IDS["index"]]
                middle_tip = hand_landmarks.landmark[TIP_IDS["middle"]]
                ex = int((index_tip.x + middle_tip.x) / 2 * w)
                ey = int((index_tip.y + middle_tip.y) / 2 * h)

                cv2.circle(frame, (ex, ey), erase_radius, (0, 0, 255), 2)

                strokes = [
                    s for s in strokes
                    if point_to_segment_dist((ex, ey), s["p1"], s["p2"]) > erase_radius
                ]
    else:
        prev_right_point = None
        prev_left_wrist = None
        prev_right_wrist = None

    # --- Terapkan pergeseran (pan) ke seluruh goresan tersimpan ---
    if drag_delta is not None and drag_delta != (0, 0):
        dx, dy = drag_delta
        for s in strokes:
            s["p1"] = (s["p1"][0] + dx, s["p1"][1] + dy)
            s["p2"] = (s["p2"][0] + dx, s["p2"][1] + dy)

    # --- Gambar ulang semua goresan tiap frame ---
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    glow_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for s in strokes:
        cv2.line(canvas, s["p1"], s["p2"], s["color"], s["thickness"])
        cv2.line(glow_canvas, s["p1"], s["p2"], s["color"], s["thickness"] + 10)

    blurred_glow = cv2.GaussianBlur(glow_canvas, (0, 0), sigmaX=9, sigmaY=9)
    combined = frame.astype(np.float32) + blurred_glow.astype(np.float32) * 0.6
    combined = np.clip(combined, 0, 255).astype(np.uint8)

    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
    combined[mask > 0] = canvas[mask > 0]

    status = f"Gambar: {'ON' if right_active else 'off'}  |  Hapus: {'ON' if left_active else 'off'}  |  Goresan: {len(strokes)}"
    cv2.putText(combined, status, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Hand Tracking - Gambar 2 Tangan", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        strokes = []

cap.release()
cv2.destroyAllWindows()
