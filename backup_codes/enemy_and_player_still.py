import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

height = 600
width = 1200
black_frame = np.zeros((height, width, 3), dtype=np.uint8)

#---------------------------enemy
enemy_frame = cv2.imread('boxer.jpg')
enemy_frame_rgb = cv2.cvtColor(enemy_frame,cv2.COLOR_BGR2RGB)
enemy_output = pose.process(enemy_frame_rgb)
if enemy_output.pose_landmarks:
    mp_drawing.draw_landmarks(
        black_frame,
        enemy_output.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )
#----------------------------player
player_frame = cv2.imread('boxer.jpg')
player_frame = cv2.flip(player_frame,1)
player_frame_rgb = cv2.cvtColor(player_frame,cv2.COLOR_BGR2RGB)
player_output = pose.process(player_frame_rgb)
if player_output.pose_landmarks:
    mp_drawing.draw_landmarks(
        black_frame,
        player_output.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )
#----------------------------------------------------------------------
cv2.imshow("VIRTUAL HIT",black_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

