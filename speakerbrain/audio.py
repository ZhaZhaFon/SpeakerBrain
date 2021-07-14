
# 音频依赖
import soundfile as sf

# 深度学习依赖
import torch

# 读取音频
def read_audio(file):
    wav, sr = sf.read(file)
    return wav

# 读取音频为Tensor
cuda = torch.device('cuda')
def read_audio_tensor(file):
    wav, sr = sf.read(file)
    return torch.Tensor(wav, device=cuda)
