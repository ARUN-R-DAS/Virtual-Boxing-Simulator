import pyautogui
import cv2
import pygame
import subprocess

pygame.mixer.init()
audio = pygame.mixer.Sound("music\intro_audio.mp3")
audio.play()

screen_width, screen_height = pyautogui.size()
print(f"screen resolution : {screen_width} x {screen_height}")

# Start preloading the game in the background
game_process = subprocess.Popen(["python", "game_final.py"])

cutscene = cv2.VideoCapture('videos\intro.mp4')

cv2.namedWindow("cutscene", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("cutscene", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    success,frame = cutscene.read()
    if not success:
        break
    cv2.imshow("cutscene",frame)
    if cv2.waitKey(21) & 0xFF in [ord('q'),ord('Q')]:
        break

cutscene.release()
# cv2.destroyAllWindows()
