import mediapipe as mp
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import pygame
import pyautogui
from utils import draw_body_shapes, get_keypoints_xy, draw_debug_shapes, is_hit


screen_width, screen_height = pyautogui.size()

pygame.mixer.init()
punch_sound = pygame.mixer.Sound("music\punch_short.mp3")

enemy_video_path = 'videos\enemy_m4.mp4'

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
player_pose = mp_pose.Pose()
enemy_pose = mp_pose.Pose()

#black frame dimensions
height = screen_height
width = screen_width

start_game = False

player_head_pos = enemy_head_pos = None
player_fist1_pos = player_fist2_pos = None
enemy_fist1_pos = enemy_fist2_pos = None

player_facing_right = False

#---------------------------main loop
player_video = cv2.VideoCapture(0)
enemy_video = cv2.VideoCapture(enemy_video_path)
while True:
    #black frame resetting
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)

    #---------------------------player
    player_success,player_frame = player_video.read()
    player_frame_flipped = cv2.flip(player_frame,1)
    player_frame_rgb = cv2.cvtColor(player_frame_flipped,cv2.COLOR_BGR2RGB)
    player_output = player_pose.process(player_frame_rgb)
    if player_output.pose_landmarks:
        draw_body_shapes(black_frame, player_output.pose_landmarks, shift_x_px=0, width=1280, height=480,color=(255,0,0))
        points_vid = get_keypoints_xy(player_output.pose_landmarks, shift_x_px=0, width=1280, height=480)
        player_head_pos = points_vid["head"]
        player_fist1_pos = points_vid["left_fist"]
        player_fist2_pos = points_vid["right_fist"]
        player_foot1_pos = points_vid["left_foot"]
        player_foot2_pos = points_vid["right_foot"]
        # draw_debug_shapes(black_frame, points_vid, color=(0, 255, 255))

        #===========================measuring distance to cam
        if not start_game:
            point1 = player_output.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            point2 = player_output.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
            # Convert normalized coordinates (0–1) to pixels
            point1_y = int(point1.y * height)
            point2_y = int(point2.y * height)
            length = abs(point1_y - point2_y)
            if length>2000: # change back after debug
                Text = "Too Close! please move back"
            elif length<260:
                Text = "Too Far! Please move close"
            else:
                Text = "Perfect Distance! Starting game"
                start_game = True
            cv2.putText(
                black_frame,
                str(length) + Text,
                (200,200),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=2,
                color=(0,255,0),
                thickness=2
            )

    #---------------------------enemy
    if start_game:
        enemy_success,enemy_frame = enemy_video.read()
        #looping video
        if not enemy_success:
            enemy_video.set(cv2.CAP_PROP_POS_FRAMES,0)
            continue
        if player_facing_right:
            enemy_frame = cv2.flip(enemy_frame,1)
        enemy_frame_rgb = cv2.cvtColor(enemy_frame,cv2.COLOR_BGR2RGB)
        enemy_output = enemy_pose.process(enemy_frame_rgb)
        if enemy_output.pose_landmarks:
            draw_body_shapes(black_frame, enemy_output.pose_landmarks, shift_x_px=0, width=1280, height=480,color=(0,0,255))
            points_vid = get_keypoints_xy(enemy_output.pose_landmarks, shift_x_px=0, width=1280, height=480)
            enemy_head_pos = points_vid["head"]
            enemy_fist1_pos = points_vid["left_fist"]
            enemy_fist2_pos = points_vid["right_fist"]
            enemy_foot1_pos = points_vid["left_foot"]
            enemy_foot2_pos = points_vid["right_foot"]
            # draw_debug_shapes(black_frame, points_vid, color=(0, 255, 255))
    #-----------------------------hit detection
    if player_head_pos and enemy_head_pos:
        for enemy_hit_points in [enemy_fist1_pos,enemy_fist2_pos,enemy_foot1_pos,enemy_foot2_pos]:
            if is_hit(enemy_hit_points, player_head_pos):
                print("Player Hit !")
                punch_sound.play()
                cv2.putText(black_frame, "Player Hit!", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)


        for player_hit_points in [player_fist1_pos,player_fist2_pos,player_foot1_pos,player_foot2_pos]:
            if is_hit(player_hit_points, enemy_head_pos):
                print("Enemy Hit !")
                punch_sound.play()
                cv2.putText(black_frame, "Enemy Hit!", (900,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    #-------------------------------flip enemy based on player pos
    if player_head_pos and enemy_head_pos:
        if player_head_pos[0] < enemy_head_pos[0]:
            player_facing_right = False
        else:
            player_facing_right = True

        # print(enemy_facing_right)
    #-----------------------------cleanup-----------------------------------------
    cv2.imshow("VIRTUAL HIT",black_frame)
    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break
player_video.release()
enemy_video.release()
cv2.destroyAllWindows()

