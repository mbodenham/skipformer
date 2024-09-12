# Skipformer: Evolving Beyond Blocks for Extensively Searching On-Device Language Models with Learnable Attention Window
Accompanying code for IEEE Access article (https://ieeexplore.ieee.org/document/10666862).

## Custom Attention Window CUDA Kernel
The kernel can be tested on Jetson Nano using the provided `test_kernel.ipynb` notebook.
The notebook code for generating Figure 4 for the paper submission and tests kernel accuracy.

Tests and Figures for Attention Window Size CUDA Kernel

## Training on OpenWebText

To train models run;
GPT-2 Small `./train_skipformer gpt2-small`
GPT2-Small-W `./train_skipformer gpt2-small-w`
Skipformer A `./train_skipformer skipformer-a`
Skipformer B `./train_skipformer skipformer-b`
