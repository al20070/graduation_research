import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize
import U_Net

# U-Netのモデル定義とデータの前処理は前回のコードを再利用

# ダミーデータで代用するためのDatasetクラスを定義
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000, image_size=(256, 256), num_classes=21):
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = torch.randn(3, *self.image_size)  # ダミーのRGB画像
        target = torch.randint(0, self.num_classes, self.image_size)  # ダミーのラベル画像
        return image, target

# ダミーデータを用いたトレーニング
def train_unet(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# ダミーデータセットを作成
train_dataset = DummyDataset(size=1000, image_size=(256, 256), num_classes=21)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# モデルの初期化（前回のコードを再利用）
in_channels = 3
out_channels = 21
model = U_Net.UNet(in_channels, out_channels)

# 損失関数とオプティマイザの定義
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# トレーニングの実行
train_unet(model, train_loader, criterion, optimizer, num_epochs=10)
