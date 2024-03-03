import torch
import torch.nn as nn
import torch.optim as optim

class BayesianMaskingModel(nn.Module):
    def __init__(self, feature_dim, kernel_size=3, padding=1):
        super(BayesianMaskingModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim,
                                kernel_size=kernel_size, padding=padding, groups=feature_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, feature, timestep)
        x = self.conv1d(x)
        x = self.sigmoid(x)  # 확률 값을 얻기 위해 시그모이드 적용
        return x

# 모델, 손실 함수, 옵티마이저 초기화
feature_dim = 4  # 예제 feature 차원
model = BayesianMaskingModel(feature_dim=feature_dim)
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 데이터 준비 (예제)
batch, timestep, feature = 2, 5, feature_dim
data = torch.randn(batch, feature, timestep)  # 모델 입력 형태에 맞춤
target_mask = torch.randint(0, 2, (batch, feature, timestep)).float()  # 실제 마스크

# 모델 훈련
model.train()
optimizer.zero_grad()
output_mask = model(data)
loss = loss_function(output_mask, target_mask)
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")
