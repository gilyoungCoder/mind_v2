import torch
import torch.nn.functional as F

def probabilistic_mask_update(mask, kernel_size=3, padding=1, alpha = 10, beta = 1.5):
    """
    mask: 입력 마스킹 텐서, 형태는 (batch, timestep, feature)
    kernel_size: 주변을 확인할 커널의 크기 (timestep 기준)
    padding: 컨볼루션 연산 시 적용할 패딩의 크기
    """
    # 컨볼루션 필터 초기화: 각 feature마다 독립적으로 적용될 수 있도록 설정
    conv_filter = torch.ones((mask.size(2), 1, kernel_size), dtype=torch.float32, device=mask.device)
    
    # 입력 mask의 형태를 (batch, channel, length)로 조정
    mask = mask.transpose(1, 2)  # 형태를 (batch, feature, timestep)로 변경
    
    # 1D 컨볼루션을 사용하여 각 timestep 주변의 1의 개수를 계산
    neighbors_count = F.conv1d(mask, conv_filter, padding=padding, groups=mask.size(1))
    # print("nc",neighbors_count)
    # 주변에 1이 있는 비율 계산
    neighbors_ratio = neighbors_count / kernel_size
    
    # 확률적 업데이트를 위한 마스크 생성
    higher_probs = torch.sigmoid(neighbors_ratio * alpha - beta)
    # 확률적 업데이트를 위한 마스크 생성
    updated_mask = torch.bernoulli(higher_probs)
    # updated_mask = torch.bernoulli(neighbors_ratio)
    # updated_mask = torch.where(neighbors_ratio >= alpha, torch.ones_like(neighbors_ratio), torch.zeros_like(neighbors_ratio))

    
    # 마스크 형태를 원래대로 복구 (batch, timestep, feature)
    updated_mask = updated_mask.transpose(1, 2)
    
    return updated_mask

# # 예제 마스킹 텐서 생성
# batch, timestep, feature = 2, 5, 4
# original_mask = torch.randint(0, 2, (batch, timestep, feature)).float()

# print("Original Mask:\n", original_mask)

# # 마스킹 텐서 업데이트
# updated_mask = probabilistic_mask_update(original_mask, kernel_size=3, padding=1)

# print("Updated Mask:\n", updated_mask)
