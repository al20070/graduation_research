#視線情報をヒートマップとして画像に重ねて出力

#動画の左右の予測がされてない気がする(中央辺りはよさそう)
#処理重い問題　＞　画質下げて処理速くする？

import cv2
from model import *
from data import *

#U-Netのモデル呼び出し
model = unet()
model.load_weights("./model_v01.hdf5")

#入力動画の読み込み
#cap = cv2.VideoCapture(0) #PCのカメラ使用
cap = cv2.VideoCapture('./img/video.mp4') #動画読み込み

#出力ウィンドウのサイズ変更用
resize = 2 #画面の縮小倍率
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/resize) #入力動画の縦/resize
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/resize) #入力動画の横/resize

i = 0

while(1):

    # 一フレーム取り出す
    ret, frame = cap.read()
    #print(ret) #TorF

    if i%5 == 0: #ノートパソコンだと処理が重すぎるので数フレームごと

        #frameの注視点をU-Netで予測(グレースケール画像)
        frame = cv2.resize(frame, (width, height))
        testGene = testGenerator2(frame)
        results = model.predict(testGene, 1, verbose=1)
        saveRes = saveResult2(results)
        gaze = cv2.resize(saveRes, (width, height))
        #print(frame.shape)
        #print(gaze.shape)

        #ヒートマップに変換
        heatmap = cv2.applyColorMap(gaze, cv2.COLORMAP_JET)
    
        #画像のサイズ確認
        assert frame.shape == heatmap.shape, "2つの画像のサイズは一致していなければならない"

        #アルファブレンディング
        alpha = 0.5
        blended = cv2.addWeighted(frame, alpha, heatmap, 1 - alpha, 0)
                     
        #結果を表示
        #cv2.imshow("frame", frame)
        #cv2.imshow("gaze", gaze)
        cv2.imshow("prediction", blended)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    i = i + 1
