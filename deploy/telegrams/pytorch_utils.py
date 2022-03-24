import torch


def send_to_cuda(device_index: int, tensor):
    if device_index is not None:
        return tensor.to(torch.device(device_index))
    else:
        return tensor.cuda()