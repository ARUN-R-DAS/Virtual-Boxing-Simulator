import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
player_pose = mp_pose.Pose()
enemy_pose = mp_pose.Pose()

#black frame dimensions
height = 600
width = 1200

#---------------------------main loop
player_video = cv2.VideoCapture('enemy_m2.mp4')
enemy_video = cv2.VideoCapture('enemy_m2.mp4')
while True:
    #black frame resetting
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    #---------------------------player
    player_success,player_frame = player_video.read()
    player_frame_rgb = cv2.cvtColor(player_frame,cv2.COLOR_BGR2RGB)
    player_output = player_pose.process(player_frame_rgb)
    if player_output.pose_landmarks:
        mp_drawing.draw_landmarks(
            black_frame,
            player_output.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
    #---------------------------enemy
    enemy_success,enemy_frame = enemy_video.read()
    enemy_frame_flipped = cv2.flip(enemy_frame,1)
    enemy_frame_rgb = cv2.cvtColor(enemy_frame_flipped,cv2.COLOR_BGR2RGB)
    enemy_output = enemy_pose.process(enemy_frame_rgb)
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

