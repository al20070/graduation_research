#PCのカメラ使えるかテスト ＞使えそう

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):
    # 一フレーム取り出す
    ret, frame = cap.read()
    print(ret) #TorF

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
