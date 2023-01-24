#include <torch/extension.h>
#include <ATen/ATen.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TENSOR(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor attention_window_matmul_cuda_forward(
  const at::Tensor A,
  const at::Tensor B,
  const at::Tensor WS,
  const at::Tensor WM,
  const bool Masked);


at::Tensor attention_window_matmul_forward(
  const at::Tensor A,  // A tensor of size (batch_size, num_heads, N, d_model)
  const at::Tensor B,  // B tensor of size (batch_size, num_heads, N, d_model)
  const at::Tensor WS, //Attetnion window size tensor of size (num_heads)
  const at::Tensor WM, //Window mask tensor of size (num_heads, N, N)
  const bool Masked    //Mask forward tokens
){
    CHECK_TENSOR(A);
    CHECK_TENSOR(B);
    CHECK_TENSOR(WS);
    CHECK_TENSOR(WM);

  return attention_window_matmul_cuda_forward(A,
                                              B,
                                              WS,
                                              WM,
                                              Masked);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &attention_window_matmul_forward, "Attention Window Matrix Multiplication Forward (CUDA)");
}
