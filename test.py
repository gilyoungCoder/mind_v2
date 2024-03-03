import torch

# 임의의 2D 텐서 생성
x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 첫 번째 차원(행)을 따라 합 계산
row_sum = torch.sum(x, dim=0)

# 두 번째 차원(열)을 따라 합 계산
col_sum = torch.sum(x, dim=1)
print(x, x.shape)
print(f"Row Sum: {row_sum}, {row_sum.shape}")
print(f"Column Sum: {col_sum}, {col_sum.shape}")
