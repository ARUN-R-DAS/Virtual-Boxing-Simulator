import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

#black frame dimensions
height = 600
width = 1200
#preprocess enemy image only once
enemy_frame = cv2.imread('boxer.png')
enemy_frame = cv2.flip(enemy_frame,1)
enemy_frame_rgb = cv2.cvtColor(enemy_frame,cv2.COLOR_BGR2RGB)
enemy_output = pose.process(enemy_frame_rgb)

#---------------------------main loop
player_video = cv2.VideoCapture('enemy_m2.mp4')
while True:
    #black frame resetting
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    #---------------------------player
    success,player_frame = player_video.read()
    player_frame_rgb = cv2.cvtColor(player_frame,cv2.COLOR_BGR2RGB)
    player_output = pose.process(player_frame_rgb)
    if player_output.pose_landmarks:
        mp_drawing.draw_landmarks(
            black_frame,
            player_output.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
    #---------------------------enemy
    if enemy_output.pose_landmarks:
        mp_drawing.draw_landmarks(
            black_frame,
            enemy_output.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    #----------------------------------------------------------------------
    cv2.imshow("VIRTUAL HIT",black_frame)
    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break
cv2.waitKey(0)
cv2.destroyAllWindows()

