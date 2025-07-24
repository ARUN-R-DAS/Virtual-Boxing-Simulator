# utils.py

import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def draw_body_shapes(image, landmarks, shift_x_px, width, height, color):
    def get_point(lm):
        return int((lm.x + shift_x_px / width) * width), int(lm.y * height)

    lm = landmarks.landmark
    def p(name): return get_point(lm[mp_pose.PoseLandmark[name].value])

    def midpoint(pt1, pt2):
        return ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)

    # Points
    nose = p("NOSE")
    ls = p("LEFT_SHOULDER")
    rs = p("RIGHT_SHOULDER")
    le = p("LEFT_ELBOW")
    re = p("RIGHT_ELBOW")
    lw = p("LEFT_WRIST")
    rw = p("RIGHT_WRIST")
    lh = p("LEFT_HIP")
    rh = p("RIGHT_HIP")
    lk = p("LEFT_KNEE")
    rk = p("RIGHT_KNEE")
    la = p("LEFT_ANKLE")
    ra = p("RIGHT_ANKLE")

    # Head
    cv2.circle(image, nose, 20, color, -1)

    # Torso
    torso_pts = np.array([ls, rs, rh, lh], np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(image, [torso_pts], color)

    # Arms
    cv2.line(image, ls, le, color, 12)
    cv2.line(image, le, lw, color, 12)
    cv2.line(image, rs, re, color, 12)
    cv2.line(image, re, rw, color, 12)

    # Legs
    cv2.line(image, lh, lk, color, 14)
    cv2.line(image, lk, la, color, 14)
    cv2.line(image, rh, rk, color, 14)
    cv2.line(image, rk, ra, color, 14)

    # Feet
    cv2.circle(image, la, 10, color, -1)
    cv2.circle(image, ra, 10, color, -1)

def get_keypoints_xy(landmarks, shift_x_px, width, height):
    def to_pixel_coords(lm):
        x = int((lm.x + shift_x_px / width) * width)
        y = int(lm.y * height)
        return x, y

    lm = landmarks.landmark
    points = {
        "head": to_pixel_coords(lm[mp_pose.PoseLandmark.NOSE]),
        "left_fist": to_pixel_coords(lm[mp_pose.PoseLandmark.LEFT_WRIST]),
        "right_fist": to_pixel_coords(lm[mp_pose.PoseLandmark.RIGHT_WRIST]),
        "left_foot": to_pixel_coords(lm[mp_pose.PoseLandmark.LEFT_ANKLE]),
        "right_foot": to_pixel_coords(lm[mp_pose.PoseLandmark.RIGHT_ANKLE])
    }
    return points

def draw_debug_shapes(image, points, color):
    cv2.circle(image, points["head"], 10, color, -1)
    for key in ["left_fist", "right_fist"]:
        x, y = points[key]
        cv2.rectangle(image, (x - 12, y - 12), (x + 12, y + 12), color, -1)
    for key in ["left_foot", "right_foot"]:
        x, y = points[key]
        cv2.circle(image, (x, y), 12, color, -1)

def is_hit(p1, p2, threshold=60):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < threshold
