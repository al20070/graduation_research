#視線情報をヒートマップとして画像に重ねて出力

import cv2
import matplotlib.pyplot as plt
import numpy as np
from model import *
from data import *

model = unet()
model.load_weights("./model_v02-27.hdf5")

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./img/video.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
i = 0

while(1):
    # 一フレーム取り出す
    ret, frame = cap.read()
    #print(ret) #TorF
    if i%5 == 0:
        frame = cv2.resize(frame, (int(width/2), int(height/2)))
        cv2.imwrite("./img/0.png", frame)

        #frameの注視点をU-Netで予測(グレースケール画像)
        DIR = "./img/" #テスト画像のあるフォルダのパス
        NUM_IMG = 1 #画像の枚数
        testGene = testGenerator(DIR, NUM_IMG)
        results = model.predict_generator(testGene, NUM_IMG, verbose=1)
        saveResult(DIR,results)
        gaze = cv2.imread(DIR + "0_predict.png")
        gaze = cv2.resize(gaze, (int(width/2), int(height/2)))
        #print(frame.shape)
        #print(gaze.shape)

        #ヒートマップに変換
        heatmap = cv2.applyColorMap(gaze, cv2.COLORMAP_JET)
        #print(heatmap.shape)
    
        #画像のサイズ確認
        assert frame.shape == heatmap.shape, "2つの画像のサイズは一致していなければならない"

        # アルファブレンディング
        alpha = 0.5
        blended = cv2.addWeighted(frame, alpha, heatmap, 1 - alpha, 0)

        # 結果を表示
        #cv2.imshow("frame", frame)
        cv2.imshow("tes", blended)
        #cv2.imshow("gaze", gaze)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    
    i = i + 1
