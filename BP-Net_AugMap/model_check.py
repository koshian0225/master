from torchinfo import summary
from models.BPNet import Net
import torch

model = Net()

# ダミー入力データを作成
I = torch.randn(1, 3, 256, 1216)  # Bx3xHxW
S = torch.randn(1, 1, 256, 1216)  # Bx1xHxW
K = torch.randn(1, 3, 3)         # Bx3x3

summary(
    model,
    input_data=(I, S, K),
    col_names=["output_size", "num_params"],
    depth=10
)