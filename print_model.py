import torch
from torchinfo import summary

try:
    from model.sscnn import SSCNN
    from model.c_net import CNET
    from model.rac_net import RACNET
    from model.rc_net import RCNET
    from model.convnext_1d import CONVNEXT1D
    from model.one_dim_cnn import ONEDIMCNN
except ImportError:
    raise ImportError('[Error] import model failed!')

if __name__ == '__main__':
    batch_size = 32
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Info] Using device {device}')
    # load model
    model = ONEDIMCNN(
        in_channel=1,
        out_channel=5,
        spectrum_size=3584).to(device)
    # print model
    summary(model, input_size=(batch_size, 1, 3584))
