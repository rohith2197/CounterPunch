import cv2
import mediapipe as mp
import numpy as np
from shapely.geometry import Polygon

# Configuration (Optimized for Guard Analysis)
ARM_THICKNESS = 150
FOREARM_EXTENSION = 0.6

# Initialize mediapipe
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1)
pose = mp_pose.Pose(static_image_mode=True)

# Helper Methods used
def poly_from_box(box):
    return Polygon([(int(p[0]), int(p[1])) for p in box])

#Calculates eprcent of overlap bewteen susceptible zones and arm polygons
def overlap_percent(target, blocker):
    if target is None or blocker is None:
        return 0.0
    inter = target.intersection(blocker)
    if inter.is_empty:
        return 0.0
    return (inter.area / target.area) * 100

# Creates hitboxes for the face and the surrounding area for susceptibilities
def face_square_and_sides(face_lm, w, h):
    xs = [int(lm.x * w) for lm in face_lm.landmark]
    ys = [int(lm.y * h) for lm in face_lm.landmark]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    size = max(max_x - min_x, max_y - min_y)
    cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2

    face_rect = ((cx, cy), (size, size), 0)
    face_box = cv2.boxPoints(face_rect).astype(int)
    face_poly = poly_from_box(face_box)

    half = size // 2
    left_rect = Polygon([
        (cx - size - half, cy - size//2),
        (cx - half,        cy - size//2),
        (cx - half,        cy + size//2),
        (cx - size - half, cy + size//2)
    ])
    right_rect = Polygon([
        (cx + half,        cy - size//2),
        (cx + size + half, cy - size//2),
        (cx + size + half, cy + size//2),
        (cx + half,        cy + size//2)
    ])
    return face_box, face_poly, left_rect, right_rect

# Creates a torso polygon to analyze susceptibility to body shots
def torso_polys(pose_lm, w, h):
    lm = pose_lm.landmark
    ls, rs = lm[11], lm[12]
    lh, rh = lm[23], lm[24]

    x1 = int(min(ls.x, rs.x) * w)
    x2 = int(max(ls.x, rs.x) * w)
    y1 = int(min(ls.y, rs.y) * h)
    y2 = int(max(lh.y, rh.y) * h)

    h_total = y2 - y1
    h_upper = int(h_total * 0.2)

    upper = Polygon([
        (x1, y1), (x2, y1),
        (x2, y1 + h_upper), (x1, y1 + h_upper)
    ])
    lower = Polygon([
        (x1, y1 + h_upper), (x2, y1 + h_upper),
        (x2, y2), (x1, y2)
    ])
    return upper, lower

# Creates rotatable arm polygons to show coverage of arms over susceptible zones
def rotated_arm(p1, p2, thickness, extend=0.0):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    p2_ext = (int(p2[0] + dx * extend), int(p2[1] + dy * extend))
    cx = (p1[0] + p2_ext[0]) // 2
    cy = (p1[1] + p2_ext[1]) // 2
    new_length = int(np.hypot(p2_ext[0] - p1[0], p2_ext[1] - p1[1]))
    angle = np.degrees(np.arctan2(dy, dx))
    rect = ((cx, cy), (new_length, thickness), angle)
    box = cv2.boxPoints(rect).astype(int)
    return box, poly_from_box(box)

# Creates polygons for arms and is used to calcualte overlap
def arm_polys(pose_lm, w, h):
    lm = pose_lm.landmark
    def p(i): return int(lm[i].x * w), int(lm[i].y * h)
    arms = []

    # Upper arms
    for a, b in [(11,13), (12,14)]:
        box, poly = rotated_arm(p(a), p(b), ARM_THICKNESS, extend=0.0)
        arms.append((box, poly))

    # Forearms with extension
    for a, b in [(13,15), (14,16)]:
        box, poly = rotated_arm(p(a), p(b), ARM_THICKNESS, extend=FOREARM_EXTENSION)
        arms.append((box, poly))

    return arms

# Takes an image and analyzes the guard protection (from arms), returns an image and guard protection values
def image_guard_analysis(image, draw=True):
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = image.copy()

    face_res = face_mesh.process(rgb)
    pose_res = pose.process(rgb)

    if not face_res.multi_face_landmarks or not pose_res.pose_landmarks:
        return output, None

    # Generates the required polygons
    face_box, face_poly, left_face, right_face = face_square_and_sides(face_res.multi_face_landmarks[0], w, h)
    upper_torso, lower_torso = torso_polys(pose_res.pose_landmarks, w, h)
    arms = arm_polys(pose_res.pose_landmarks, w, h)

    # Calculates the overlap between arm polygons and susceptibility polygons
    face_cov = left_cov = right_cov = uppercut_cov = body_cov = 0
    for _, arm in arms:
        face_cov += overlap_percent(face_poly, arm)
        left_cov += overlap_percent(left_face, arm)
        right_cov += overlap_percent(right_face, arm)
        uppercut_cov += overlap_percent(upper_torso, arm)
        body_cov += overlap_percent(lower_torso, arm)

    jab = min(face_cov, 100) > 3
    left_hook = min(left_cov, 100) > 6
    right_hook = min(right_cov, 100) > 6
    uppercut = min(uppercut_cov, 100) > 30
    body_shot = min(body_cov, 100) > 20

    if draw:
        # Draw face rectangle
        cv2.polylines(output, [face_box], True, (0,255,0) if jab else (0,0,255), 2)
        # Draw left/right face sides
        cv2.polylines(output, [np.array(left_face.exterior.coords, int)], True, (0,255,0) if left_hook else (0,0,255), 2)
        cv2.polylines(output, [np.array(right_face.exterior.coords, int)], True, (0,255,0) if right_hook else (0,0,255), 2)
        # Draw torso
        cv2.polylines(output, [np.array(upper_torso.exterior.coords, int)], True, (0,255,0) if uppercut else (0,0,255), 2)
        cv2.polylines(output, [np.array(lower_torso.exterior.coords, int)], True, (0,255,0) if body_shot else (0,0,255), 2)
        # Draw arms
        for box, _ in arms:
            cv2.polylines(output, [box], True, (255,0,0), 2)

    return output, (jab, left_hook, right_hook, uppercut, body_shot)
