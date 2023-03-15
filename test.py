import torch
from retro_pytorch import RETRO

device = "cuda"

retro = RETRO(
    chunk_size=64,  # the chunk size that is indexed and retrieved (needed for proper relative positions as well as causal chunked cross attention)
    max_seq_len=2048,  # max sequence length
    enc_dim=896,  # encoder model dim
    enc_depth=2,  # encoder depth
    dec_dim=796,  # decoder model dim
    dec_depth=12,  # decoder depth
    dec_cross_attn_layers=(3, 6, 9, 12),  # decoder cross attention layers (with causal chunk cross attention)
    heads=8,  # attention heads
    dim_head=64,  # dimension per head
    dec_attn_dropout=0.25,  # decoder attention dropout
    dec_ff_dropout=0.25,  # decoder feedforward dropout
    use_deepnet=True  # turn on post-normalization with DeepNet residual scaling and initialization, for scaling to 1000 layers
)

if device == "cuda":
    retro.to(torch.device("cuda"))

# randint(low, high=None, size=None, dtype='l')
# [0,20000]数值范围，token_id，2是batch_size表示有两个训练样本，2048是输入长度，1是等待预测的token_id。
seq = torch.randint(0, 20000, (20, 512 + 1), device=device)
# [0,20000]数值范围，token_id，2是因为有两个训练训练样本
# 32是因为每一个输入样本可以划分出这么多chunk，每一个输入chunk有2个邻居
# 2表示每一个输入chunk有2个邻居
# 128是64*2，前半计算key，后半是其的后继
retrieved = torch.randint(0, 20000, (20, 8, 2, 128), device=device)  # retrieved tokens - (batch, num chunks, num retrieved neighbors, retrieved chunk with continuation)

for i in range(100):
    loss = retro(seq, retrieved, return_loss=True)
    print(loss)
    loss.backward()
