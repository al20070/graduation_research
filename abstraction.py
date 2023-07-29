import cv2
import numpy as np

cap = cv2.VideoCapture('./video/video4.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("cap(h,w) : ", height, width)

#青を抽出
bgr = [255,200,34]
thresh = 34
#色の閾値
minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])

while(1):
    # 一フレーム取り出す
    ret, frame = cap.read()
    #print(ret) #TorF

    #動画のサイズ変更
    #frame = cv2.resize(frame, (int(width/2), int(height/2)))
    #ここまで

	#画像のマスク
    mask1 = cv2.inRange(frame, minBGR, maxBGR)
    #print("mask : ", mask.shape) #(h,w)
    #画像のマスク（合成）
    result = cv2.bitwise_and(frame, frame, mask = mask1)
    
	#グレースケール
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    gray_3ch = np.stack((gray,)*3, -1)
	
    # 閾値の設定
    threshold = 10
    # 二値化(閾値を超えた画素を255にする。)
    ret, thresh = cv2.threshold(gray_3ch, threshold, 255, cv2.THRESH_BINARY)

    #ガウシアンフィルタ
    blur = cv2.GaussianBlur(thresh,(75,75),0)
    #print("blur : ", blur.shape) #(h,w)

    #cv2.imshow("frame", frame)       #元
    #cv2.imshow("Result BGR", result) #マスクした動画
    #cv2.imshow("Result mask", mask)  #マスク部分
    #cv2.imshow('mask',blur)          #ガウシアンフィルタ
    
    mergeImg = np.hstack((frame, blur))  #元動画+マスクをかけた動画
    cv2.imshow("result", mergeImg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
