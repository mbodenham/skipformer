import torch
from torch.utils.cpp_extension import load
import os

if not os.path.exists(os.path.join(os.path.dirname(__file__), 'build')):
    os.makedirs(os.path.join(os.path.dirname(__file__), 'build'))

awmm_cuda = load(name='attention_window_matmul',
                 sources=[os.path.join(os.path.dirname(__file__), 'attention_window_matmul.cpp'),
                          os.path.join(os.path.dirname(__file__), 'attention_window_matmul_kernel.cu')],
                 build_directory=os.path.join(os.path.dirname(__file__), 'build'),
                 verbose=True)

class attention_window_matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, window_size, ramp_mask, masked=False):
        assert isinstance(window_size, torch.Tensor)
        assert isinstance(masked, bool)
        assert len(q.shape) == 4
        assert len(k.shape) == 4
        assert window_size.shape[0] == q.shape[1]

        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()

        out = awmm_cuda.forward(q,
                                k,
                                window_size,
                                ramp_mask,
                                masked)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
        return None, None, None, None, None
