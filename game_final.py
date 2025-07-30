#game final.py

import mediapipe as mp
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import pygame
import pyautogui
import time
from utils import draw_body_shapes, get_keypoints_xy, is_hit, return_player_height, draw_health_bar

def run_game():
    pygame.mixer.init()
    audio2 = pygame.mixer.Sound("music\level_music.mp3")
    punch_sound = pygame.mixer.Sound(r"music\punch_short.mp3")
    audio2.play()

    screen_width, screen_height = pyautogui.size()

    width,height = 1260,480

    enemy_video_path = r'videos\enemy_m4.mp4'

    mp_draw = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    player_pose = mp_pose.Pose()
    enemy_pose = mp_pose.Pose()

    start_game = False
    player_head_pos = enemy_head_pos = None
    player_fist1_pos = player_fist2_pos = None
    enemy_fist1_pos = enemy_fist2_pos = None
    player_facing_right = False

    # Cooldown settings (in seconds)
    hit_cooldown = .3
    last_player_hit_time = 0
    last_enemy_hit_time = 0

    player_health = 100
    enemy_health = 100
    damage_per_hit = 5

    cv2.namedWindow("game", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    #---------------------------main loop
    player_video = cv2.VideoCapture(0)
    enemy_video = cv2.VideoCapture(enemy_video_path)
    while True:
        #black frame resetting
        black_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

        #---------------------------player
        player_success,player_frame = player_video.read()
        player_frame_flipped = cv2.flip(player_frame,1)
        player_frame_rgb = cv2.cvtColor(player_frame_flipped,cv2.COLOR_BGR2RGB)
        player_output = player_pose.process(player_frame_rgb)
        # Draw webcam feed as picture-in-picture
        if player_success:
            webcam_preview = cv2.resize(player_frame_flipped, (160, 120))  # Resize small
            y1 = screen_height - 130  # 10 pixels above bottom
            y2 = screen_height - 10
            x1 = screen_width - 170  # 10 pixels from right edge
            x2 = screen_width - 10
            black_frame[y1:y2, x1:x2] = webcam_preview
            
        if player_output.pose_landmarks:
            draw_body_shapes(black_frame, player_output.pose_landmarks, width=width, height=height,color=(255,0,0))
            points_vid = get_keypoints_xy(player_output.pose_landmarks, width=width, height=height)
            player_head_pos = points_vid["head"]
            player_fist1_pos = points_vid["left_fist"]
            player_fist2_pos = points_vid["right_fist"]
            player_foot1_pos = points_vid["left_foot"]
            player_foot2_pos = points_vid["right_foot"]

            #===========================measuring player height to start the game
            player_height = return_player_height(player_output, screen_height, black_frame)
            if not start_game:
                if 600>player_height>500:
                    for countdown in range(5,1,-1):
                        black_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
                        cv2.putText(
                            black_frame,
                            "starting game in "+str(countdown),
                            (100,100),
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=1,
                            color=(0,255,0),
                            thickness=1
                        )
                        cv2.namedWindow("gametimer", cv2.WND_PROP_FULLSCREEN)
                        cv2.setWindowProperty("gametimer", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        cv2.imshow("gametimer",black_frame)
                        cv2.waitKey(1)
                        time.sleep(1)
                    
                    else:
                        start_game = True
                    cv2.destroyWindow("gametimer")
                else:
                    if player_height>600:
                        Text = "Too close! Move back from cam"
                    elif player_height<500:
                        Text = "Too far! Move closer to cam"
                    cv2.putText(
                        black_frame,
                        str(player_height)+Text,
                        (100,100),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=1,
                        color=(0,255,0),
                        thickness=1
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
                draw_body_shapes(black_frame, enemy_output.pose_landmarks, width=width, height=height, color=(0,0,255))
                points_vid = get_keypoints_xy(enemy_output.pose_landmarks, width=width, height=height)
                enemy_head_pos = points_vid["head"]
                enemy_fist1_pos = points_vid["left_fist"]
                enemy_fist2_pos = points_vid["right_fist"]
                enemy_foot1_pos = points_vid["left_foot"]
                enemy_foot2_pos = points_vid["right_foot"]
        
        #-----------------------------hit detection
        current_time = time.time()
        if player_head_pos and enemy_head_pos:
            for enemy_hit_points in [enemy_fist1_pos,enemy_fist2_pos,enemy_foot1_pos,enemy_foot2_pos]:
                if is_hit(enemy_hit_points, player_head_pos):
                    if current_time - last_player_hit_time > hit_cooldown:
                        print("Player Hit !")
                        player_health = max(0, player_health-damage_per_hit)
                        punch_sound.play()
                        cv2.putText(black_frame, "Player Hit!", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                        last_player_hit_time = current_time
                    break

            for player_hit_points in [player_fist1_pos,player_fist2_pos,player_foot1_pos,player_foot2_pos]:
                if is_hit(player_hit_points, enemy_head_pos):
                    if current_time - last_enemy_hit_time > hit_cooldown: 
                        print("Enemy Hit !")
                        enemy_health = max(0, enemy_health-damage_per_hit)
                        punch_sound.play()
                        cv2.putText(black_frame, "Enemy Hit!", (900,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                        last_enemy_hit_time = current_time
                    break
        #-------------------------------flip enemy based on player pos
        if player_head_pos and enemy_head_pos:
            if player_head_pos[0] < enemy_head_pos[0]: # x cordinate
                player_facing_right = False
            else:
                player_facing_right = True

            # Draw player and enemy health bars
            draw_health_bar(black_frame, 50, 50, 300, 25, player_health, 100, (50,50,50), (0,255,0), "Player")
            draw_health_bar(black_frame, screen_width - 350, 50, 300, 25, enemy_health, 100, (50,50,50), (0,0,255), "Enemy")
        #-------------------------------Game End
        if player_health <= 0:
            black_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            cv2.putText(black_frame, "You Lost!", (screen_width//2 - 150, screen_height//2), 
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 4)
            cv2.imshow("game", black_frame)
            cv2.waitKey(3000)
            break

        if enemy_health <= 0:
            black_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            cv2.putText(black_frame, "You Win!", (screen_width//2 - 150, screen_height//2), 
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 4)
            cv2.imshow("game", black_frame)
            cv2.waitKey(3000)
            break

        #-----------------------------cleanup-----------------------------------------
        cv2.imshow("game",black_frame)
        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break
    player_video.release()
    enemy_video.release()
    cv2.destroyAllWindows()

