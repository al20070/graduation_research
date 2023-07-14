#動画内のバブル(視聴者の視線が集中する場所)を求める
#
#動画のサイズはresizeで1920×1080にする
#結果がウィンドウに収まらない、処理が遅い場合はテンプレート画像と動画サイズを小さくする

import cv2
import numpy as np

cap = cv2.VideoCapture('./video.mkv')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while(1):
    # 一フレーム取り出す
    _, frame = cap.read()

    #動画のサイズ変更
    frame = cv2.resize(frame, (int(width/2), int(height/2)))
    #ここまで

    # BGR空間から HSV空間に変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # HSV空間で青色の範囲を定義
    hsv_min = np.array([92, 45, 70])
    hsv_max = np.array([100, 255, 255])

    # HSVイメージから青い物体だけを取り出すための閾値
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    # ビットごとのAND演算で元画像をマスク
    bit_mask = cv2.bitwise_and(frame, frame, mask = mask)

    #ノイズ除去
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (15, 15))
    erode = cv2.erode(bit_mask, kernel)
    dilate = cv2.dilate(erode, kernel)

    blur = cv2.GaussianBlur(mask,(25,25),0)

    #結果動画表示
    #cv2.imshow('frame',frame)               #元動画
    cv2.imshow('frame',blur)               #元動画
    #cv2.imshow('mask',mask)                #マスクに対応する部分表示(2値) 
    #cv2.imshow('result',bit_mask)          #マスクをかけた動画
    #cv2.imshow('result -noizu',dilate)     #ノイズ除去後の動画    
    #mergeImg = np.vstack((frame, dilate))  #元動画+マスクをかけた動画
    #cv2.imshow("result", mergeImg)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
cap.release()
mask.release()
bit_mask.release()
erode.release()
dilate.release()

cv2.destroyAllWindows()
