import cv2
import mediapipe as mp
from collections import deque

# Constant configuration for head movement analysis
PRED_LOOKAHEAD = 6

_face_mesh = None

# Initializes face mesh for head movement analysis
def init_face_mesh():
    global _face_mesh
    if _face_mesh is None:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1
        )
    return _face_mesh

# Creates a circle over the face that represents a hitbox
def get_face_circle(face_landmarks, w, h):
    xs = [int(lm.x * w) for lm in face_landmarks.landmark]
    ys = [int(lm.y * h) for lm in face_landmarks.landmark]
    cx = int(sum(xs) / len(xs))
    cy = int(sum(ys) / len(ys))
    r = int(max(max(xs) - min(xs), max(ys) - min(ys)) / 2)
    return cx, cy, r

# Predicts motion using heuristics and previous history
def motion_predict(history):
    if len(history) < 3:
        return history[-1]
    x1, y1 = history[-3]
    x2, y2 = history[-2]
    x3, y3 = history[-1]
    vx = x3 - x2
    vy = y3 - y2
    ax = x3 - 2*x2 + x1
    ay = y3 - 2*y2 + y1
    px = int(x3 + (vx + 0.5 * ax) * PRED_LOOKAHEAD)
    py = int(y3 + (vy + 0.5 * ay) * PRED_LOOKAHEAD)
    return px, py

# Creates a prediction score that ensures one dimensional head movement is penalized
def prediction_score(cx, cy, r, px, py):
    if (px - cx)**2 + (py - cy)**2 <= r**2:
        return 1.0
    if cx - r <= px <= cx + r and cy - r <= py <= cy + r:
        return 0.5
    return 0.0

# Creates the necessary drawings over the frame
def draw_head_circle(frame, cx, cy, r):
    cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)

def draw_prediction_plus(frame, px, py, r):
    s = max(5, r // 4)
    cv2.line(frame, (px - s, py), (px + s, py), (0, 0, 255), 2)
    cv2.line(frame, (px, py - s), (px, py + s), (0, 0, 255), 2)

# Creates a single function that takes history and a frame to output a head movement score,
#       a new history, and a frame with drawings on it
def analyze_frame(frame, history=None):
    if history is None:
        history = deque(maxlen=8)

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_mesh = init_face_mesh()
    results = face_mesh.process(rgb)

    output_frame = frame.copy()
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0]
        cx, cy, r = get_face_circle(lm, w, h)
        r //= 2  # scale down

        history.append((cx, cy))
        px, py = motion_predict(history) if len(history) >= 3 else (cx, cy)

        score = prediction_score(cx, cy, r, px, py)

        draw_head_circle(output_frame, cx, cy, r)
        draw_prediction_plus(output_frame, px, py, r)
    else:
        score = 0  # no face detected

    return score, history, output_frame
