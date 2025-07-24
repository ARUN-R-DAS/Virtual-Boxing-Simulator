import pyautogui
import cv2
import pygame
from game_final import run_game

pygame.mixer.init()
audio = pygame.mixer.Sound("music\intro_audio.mp3")
loading_audio = pygame.mixer.Sound("music\elevator_music.mp3")
audio.play()


screen_width, screen_height = pyautogui.size()
print(f"screen resolution : {screen_width} x {screen_height}")

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
cv2.destroyWindow('cutscene')

loading = cv2.VideoCapture('videos/loading_short.mp4')
cv2.namedWindow("loading", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("loading", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
audio.stop()
loading_audio.play()
while True:
    success, frame = loading.read()
    if not success:
        break

    cv2.imshow("loading", frame)
    if cv2.waitKey(25) & 0xFF in [ord('q'), ord('Q')]:
        break

loading.release()
cv2.destroyWindow("loading")
loading_audio.stop()
run_game()