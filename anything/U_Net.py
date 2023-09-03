import torch
import torch.nn as nn
import torch.nn.functional as F

# U-Netのエンコーダーブロック
class UNetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(UNetEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x, self.pool(x)

# U-Netのデコーダーブロック
class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(UNetDecoderBlock, self).__init__()
        self.up_sampling = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x, skip_connection):
        x = self.up_sampling(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

# U-Netモデルの定義
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = UNetEncoderBlock(in_channels, 64)
        self.encoder2 = UNetEncoderBlock(64, 128)
        self.encoder3 = UNetEncoderBlock(128, 256)
        self.encoder4 = UNetEncoderBlock(256, 512)
        self.center = UNetEncoderBlock(512, 1024)

        self.decoder4 = UNetDecoderBlock(1024, 512)
        self.decoder3 = UNetDecoderBlock(512, 256)
        self.decoder2 = UNetDecoderBlock(256, 128)
        self.decoder1 = UNetDecoderBlock(128, 64)

        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # エンコーダーブロックの順伝搬
        enc1, x = self.encoder1(x)
        enc2, x = self.encoder2(x)
        enc3, x = self.encoder3(x)
        enc4, x = self.encoder4(x)
        x, _ = self.center(x)

        # デコーダーブロックの順伝搬
        x = self.decoder4(x, enc4)
        x = self.decoder3(x, enc3)
        x = self.decoder2(x, enc2)
        x = self.decoder1(x, enc1)

        # 出力層の順伝搬
        x = self.output_conv(x)
        return x

# モデルの初期化
in_channels = 3  # 入力画像のチャンネル数（カラー画像の場合は3）
out_channels = num_classes  # 出力クラス数（セグメンテーションタスクの場合はカテゴリ数）
model = UNet(in_channels, out_channels)

# モデルの使用例（ダミーデータを用いて順伝搬のテスト）
dummy_input = torch.randn(1, in_channels, 256, 256)  # バッチサイズ1のダミー入力データ
output = model(dummy_input)
print("Output shape:", output.shape)
