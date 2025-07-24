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

start_game = False
#---------------------------main loop
player_video = cv2.VideoCapture(0)
enemy_video = cv2.VideoCapture('enemy_m3_sped.mp4')
while True:
    #black frame resetting
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)

    #---------------------------player
    player_success,player_frame = player_video.read()
    player_frame_flipped = cv2.flip(player_frame,1)
    player_frame_rgb = cv2.cvtColor(player_frame_flipped,cv2.COLOR_BGR2RGB)
    player_output = player_pose.process(player_frame_rgb)
    if player_output.pose_landmarks:
        mp_drawing.draw_landmarks(
            black_frame,
            player_output.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
        #===========================measuring distance to cam
        if not start_game:
            right_shoulder = player_output.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = player_output.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            # Convert normalized coordinates (0–1) to pixels
            right_shoulder_y = int(right_shoulder.y * height)
            right_elbow_y = int(right_elbow.y * height)
            length = abs(right_shoulder_y - right_elbow_y)
            if length>58:
                Text = "Too Close! please move back"
            elif length<50:
                Text = "Too Far! Please move close"
            else:
                Text = "Perfect Distance! Starting game"
                start_game = True
            cv2.putText(
                black_frame,
                Text,
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

