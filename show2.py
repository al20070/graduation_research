#視線情報をヒートマップとして画像に重ねて出力

#動画の左右の予測がされてない気がする(中央辺りはよさそう)
#処理重い問題　＞　画質下げて処理速くする？

import cv2
from model import *
from data import *

#U-Netのモデル呼び出し
model = unet()
model.load_weights("./model_v05-10.hdf5")

#入力動画の読み込み
img = cv2.imread("./img/img.png")

#出力ウィンドウのサイズ変更用
resize = 2 #画面の縮小倍率
width_resize  = int(img.shape[1]/resize) #入力動画の縦/resize
height_resize = int(img.shape[0]/resize) #入力動画の横/resize


#frameの注視点をU-Netで予測(グレースケール画像)
img = cv2.resize(img, (width_resize, height_resize))
testGene = testGenerator2(img)
results = model.predict(testGene, 1, verbose=1)
saveRes = saveResult2(results)
gaze = cv2.resize(saveRes, (width_resize, height_resize))
#print(frame.shape)
#print(gaze.shape)

#min...max -> 0...255　で見やすく
mi, ma = np.min(gaze), np.max(gaze)
gaze2 = (255.0 * (gaze - mi) / (ma-mi)).astype(np.uint8)        
gaze2 = cv2.equalizeHist(gaze) # ヒストグラム平坦化

#ヒートマップに変換
heatmap = cv2.applyColorMap(gaze, cv2.COLORMAP_JET)
heatmap2 = cv2.applyColorMap(gaze2, cv2.COLORMAP_JET)
 
#画像のサイズ確認
assert img.shape == heatmap.shape, "2つの画像のサイズは一致していなければならない"
assert img.shape == heatmap2.shape, "2つの画像のサイズは一致していなければならない"

#アルファブレンディング
alpha = 0.5
blended = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)
blended2 = cv2.addWeighted(img, alpha, heatmap2, 1 - alpha, 0)
        
#結果を表示
cv2.imshow("prediction1", blended)
cv2.imshow("prediction2", blended2)

cv2.waitKey(0)