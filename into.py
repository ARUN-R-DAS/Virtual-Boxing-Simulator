import pyautogui
import cv2

screen_width, screen_height = pyautogui.size()
print(f"screen resolution : {screen_width} x {screen_height}")

cutscene = cv2.VideoCapture('videos\intro.mp4')

while True:
    success,frame = cutscene.read()
    if not success:
        break
    cv2.imshow("cutscene",frame)
    if cv2.waitKey(10) & 0xFF in [ord('q'),ord('Q')]:
        break

cutscene.release()
cv2.destroyAllWindows()