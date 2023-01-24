//Useful docs https://docs.nvidia.com/cuda/cuda-math-api/index.html
// https://developer.nvidia.com/blog/boosting-productivity-and-performance-with-the-nvidia-cuda-11-2-c-compiler/
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cmath>

#define _VOLATILE_
#define load_global(x)   __ldcg(x) //Generates a `ld.global.cg` load instruction - load(ld) to gobal memory, Use ld.cg to cache loads only globally, bypassing the L1 cache, and cache only in the L2 cache.
#define assume(x)      __builtin_expect(!!(x), 1) //Complier optimization, https://tbrindus.ca/how-builtin-expect-works/

typedef struct __builtin_align__(32) {
  float s0, s1, s2, s3, s4, s5, s6, s7;
} _float8;

typedef union {
  _float8 f8;
  float val[8];
} float8x8;

__device__ void init_c_cache(
  float8x8 c_cache[8]){

  #pragma unroll
  for ( int i = 0; i < 8; i++ ){
    #pragma unroll
    for ( int j = 0; j < 8; j++ ){
      c_cache[i].val[j] = 0.f;
    }
  }
}

__device__ void matmul_thread(
  _VOLATILE_ float a_shared[8][128+4],
  _VOLATILE_ float b_shared[8][128+4],
  float8x8 c_cache[8],
  int vx, int vy){

  float a_cache[8];

  #pragma unroll
  for ( int ki = 0; ki < 8; ki++ ){

    #pragma unroll
    for ( int mi = 0; mi < 8; mi++ ){
      a_cache[mi] = a_shared[ki][8*vy + mi];
    }

    #pragma unroll
    for ( int ni = 0; ni < 8; ni++ ){
      float b = b_shared[ki][vx/4 + 8*vx + ni];

      #pragma unroll
      for ( int mi = 0; mi < 8; mi++ ){
        c_cache[mi].val[ni] = fmaf(a_cache[mi], b, c_cache[mi].val[ni]); //Compute a*b+c as a single operation, https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g5910ee832dab4f5d37118e0a6811c195
      }

    }

  }
}


// Full (no foward mask) attention kernel
__global__ void not_masked_kernel(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ C,
  const float* __restrict__ WS,
  const int M, const int N, const int K,  const int BS,  const int NH){

  int thread_id = threadIdx.x;
  int block_id = blockIdx.z;

  int batch = block_id / NH; // int / int equivalent to floor divide
  int head  = block_id - batch*NH;

  int px = blockIdx.x % 4;
  int py = blockIdx.x / 4;
  int block_dim_x = ( N + ( 128*4 ) - 1 ) / ( 128*4 );
  int block_idx_x = ( blockIdx.y % block_dim_x ) * 4 + px;
  int block_idx_y = ( blockIdx.y / block_dim_x ) * 4 + py;
  int block_start_x = block_idx_x * 128;   // starting index of block on N axis
  int block_start_y = block_idx_y * 128;   // starting index of block on M axis

  if ( block_start_x > N || block_start_y > M ){
    return;
  }

  float ws = WS[head];
  float ws_n = ws/N;
  int block_WS = ceil( (N/128) * ws_n ) + 1;
  bool do_block  = !( (block_idx_y  <= block_idx_x - block_WS)  || block_idx_y  >= block_idx_x + block_WS );
  if ( !do_block ){ // skip thread is not in attention window block
    return;
  }

  int vx = thread_id % 16;
  int vy = thread_id / 16;
  int wx = thread_id % 32; // thread idx in warp
  int wy = thread_id / 32; // warp id
  int dx = thread_id % 8;
  int dy = thread_id / 8;

  // check if thread is in the attention window
  int thread_WS = ceil( (N/8) * ws_n ) + 1;
  int thread_row = block_idx_y*16 + vy;
  int thread_col = block_idx_x*16 + vx;
  bool do_thread = !( (thread_row  <= thread_col - thread_WS) || thread_row  >= thread_col + thread_WS);

  //allocate buffer and cache
  float a_buffer0[4];
  float b_buffer0[4];
  float a_buffer1[4];
  float b_buffer1[4];
  __shared__ _VOLATILE_ float a_shared0[8][128+4];
  __shared__ _VOLATILE_ float b_shared0[8][128+4];
  __shared__ _VOLATILE_ float a_shared1[8][128+4];
  __shared__ _VOLATILE_ float b_shared1[8][128+4];

  float8x8 c_cache[8];
  init_c_cache(c_cache);

  // load 16 x 128 tile of A and B to buffer0 and buffer1
  #pragma unroll
  for ( int i = 0; i < 4; i++ ){
    int iM = block_start_y + dy + i*32;
    int iN = block_start_x + wx + i*32;

    if ( assume(iM < M) ){
      if ( assume(dx < K) ){
        a_buffer0[i] = load_global(A + (block_id)*M*K + (iM)*K + (dx));
      } else {
        a_buffer0[i] = 0.f;
      }

      if ( assume(dx+8 < K) ){
        a_buffer1[i] = load_global(A + (block_id)*M*K + (iM)*K + (dx+8));
      } else {
        a_buffer1[i] = 0.f;
      }
    }

    if ( assume(iN < N) ){
      if ( assume(wy < K) ){
        b_buffer0[i] = load_global(B + (block_id)*N*K + (wy)*N + (iN));
      } else {
        b_buffer0[i] = 0.f;
      }

      if ( assume(wy+8 < K) ){
        b_buffer1[i] = load_global(B + (block_id)*N*K + (wy+8)*N + (iN));
      } else {
        b_buffer1[i] = 0.f;
      }
    }
  }

  int nIt = (K + 16 - 1) / 16;
  #pragma unroll
  for ( int itr = 0; itr < nIt; itr++ ){
    int gStartk = itr * 16;
    int iKA = gStartk + 16 + dx;
    int iKB = gStartk + 16 + wy;

    #pragma unroll
    for ( int i = 0; i < 4; i++){
      // copy buffered tiles into shared memory
      a_shared0[dx][dy+i*32] = a_buffer0[i];
      b_shared0[wy][wx+i*32+i] = b_buffer0[i];
      a_shared1[dx][dy+i*32] = a_buffer1[i];
      b_shared1[wy][wx+i*32+i] = b_buffer1[i];

      // load next 16*128 tile of A and B to buffer0 and buffer1
      // don't load on last iteration
      if ( assume(itr < nIt - 1) ){
        int iM = block_start_y + i*32 + dy;
        int iN = block_start_x + i*32 + wx;

        if ( assume(iM < M) ){
          if ( assume(iKA < K) ){
            a_buffer0[i] = load_global(A + (block_id)*M*K + (iM)*K + (iKA));
          } else {
            a_buffer0[i] = 0.f;
          }

          if ( assume(iKA+8 < K) ){
            a_buffer1[i] = load_global(A + (block_id)*M*K + (iM)*K + (iKA+8));
          } else {
            a_buffer1[i] = 0.f;
          }
        }

        if ( assume(iN < N) ){
          if ( assume(iKB < K) ){
            b_buffer0[i] = load_global(B + (block_id)*N*K + (iKB)*N + (iN));
          } else {
            b_buffer0[i] = 0.f;
          }

          if ( assume(iKB+8 < K) ){
            b_buffer1[i] = load_global(B + (block_id)*N*K + (iKB+8)*N + (iN));
          } else {
            b_buffer1[i] = 0.f;
          }
        }
      }
    }
    // sync data loading between threads
    __syncthreads();

    // if thread is in window size compute matmul
    if ( do_thread ){
      matmul_thread(a_shared0, b_shared0, c_cache, vx, vy);
      matmul_thread(a_shared1, b_shared1, c_cache, vx, vy);
    }
    // sync matmul computation between threads
    __syncthreads();
  }

  //write cache to C
  __shared__ volatile float c_shared[16][128];
  #pragma unroll
  for ( int mi = 0; mi < 8; mi++ ){
    int iM = block_start_y + vy*8 + mi;
    // write c_cache (register memory) to c_shared (shared memory)
    if ( iM < M ){
      #pragma unroll
      for ( int ni = 0; ni < 8; ni++ ){
        int iN = block_start_x + vx*8 + ni;
        if ( assume( (iM <= iN -  ws ) || (iM  >= iN +  ws) ) ){
          c_shared[vy][vx*8 + ni] = 0.f;
        }else{
          c_shared[vy][vx*8 + ni] = c_cache[mi].val[ni];
        }
      }
      // write c_shared (shared memory) to C (global memory)
      #pragma unroll
      for ( int ni = 0; ni < 8; ni++ ){
        int iN = block_start_x + 16*ni + vx;
        if ( assume( iN < N ) ){
          C[(block_id)*M*N + (iM)*N + (iN)] = c_shared[vy][16*ni + vx];
        }
      }
    }
  }

}

__global__ void masked_kernel(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ C,
  const float* __restrict__ WS,
  const int M, const int N, const int K,  const int BS,  const int NH){

  int thread_id = threadIdx.x;
  int block_id = blockIdx.z;

  int batch = block_id / NH; // int / int equivalent to floor divide
  int head  = block_id - batch*NH;

  int px = blockIdx.x % 4;
  int py = blockIdx.x / 4;
  int block_dim_x = ( N + ( 128*4 ) - 1 ) / ( 128*4 );
  int block_idx_x = ( blockIdx.y % block_dim_x ) * 4 + px;
  int block_idx_y = ( blockIdx.y / block_dim_x ) * 4 + py;
  int block_start_x = block_idx_x * 128;   // starting index of block on N axis
  int block_start_y = block_idx_y * 128;   // starting index of block on M axis

  if ( block_start_x > N || block_start_y > M ){
    return;
  }

  float ws = WS[head];
  float ws_n = ws/N;
  int block_WS = ceil( (N/128) * ws_n ) + 1;
  bool do_block  = !( (block_idx_y < block_idx_x ) || block_idx_y  >= block_idx_x + block_WS );
  if ( assume( !do_block ) ){ // skip thread is not in attention window block
    return;
  }

  int vx = thread_id % 16;
  int vy = thread_id / 16;
  int wx = thread_id % 32; // thread idx in warp
  int wy = thread_id / 32; // warp id
  int dx = thread_id % 8;
  int dy = thread_id / 8;

  // check if thread is in the attention window
  int thread_row = block_idx_y*16 + vy;
  int thread_col = block_idx_x*16 + vx;
  int thread_WS = ceil( (N/8) * ws_n ) + 1;
  bool do_thread = !( (thread_row < thread_col ) || thread_row  >= thread_col + thread_WS );

  //allocate buffer and cache
  float a_buffer0[4];
  float b_buffer0[4];
  float a_buffer1[4];
  float b_buffer1[4];
  __shared__ _VOLATILE_ float a_shared0[8][128+4];
  __shared__ _VOLATILE_ float b_shared0[8][128+4];
  __shared__ _VOLATILE_ float a_shared1[8][128+4];
  __shared__ _VOLATILE_ float b_shared1[8][128+4];

  float8x8 c_cache[8];
  init_c_cache(c_cache);

  // load 16 x 128 tile of A and B to buffer0 and buffer1
  #pragma unroll
  for ( int i = 0; i < 4; i++ ){
    int iM = block_start_y + dy + i*32;
    int iN = block_start_x + wx + i*32;

    if ( assume(iM < M) ){
      if ( assume(dx < K) ){
        a_buffer0[i] = load_global(A + (block_id)*M*K + (iM)*K + (dx));
      } else {
        a_buffer0[i] = 0.f;
      }

      if ( assume(dx+8 < K) ){
        a_buffer1[i] = load_global(A + (block_id)*M*K + (iM)*K + (dx+8));
      } else {
        a_buffer1[i] = 0.f;
      }
    }

    if ( assume(iN < N) ){
      if ( assume(wy < K) ){
        b_buffer0[i] = load_global(B + (block_id)*N*K + (wy)*N + (iN));
      } else {
        b_buffer0[i] = 0.f;
      }

      if ( assume(wy+8 < K) ){
        b_buffer1[i] = load_global(B + (block_id)*N*K + (wy+8)*N + (iN));
      } else {
        b_buffer1[i] = 0.f;
      }
    }
  }

  int nIt = (K + 16 - 1) / 16;
  #pragma unroll
  for ( int itr = 0; itr < nIt; itr++ ){
    int gStartk = itr * 16;
    int iKA = gStartk + 16 + dx;
    int iKB = gStartk + 16 + wy;

    #pragma unroll
    for ( int i = 0; i < 4; i++){
      // copy buffered tiles into shared memory
      a_shared0[dx][dy+i*32] = a_buffer0[i];
      b_shared0[wy][wx+i*32+i] = b_buffer0[i];
      a_shared1[dx][dy+i*32] = a_buffer1[i];
      b_shared1[wy][wx+i*32+i] = b_buffer1[i];

      // load next 16*128 tile of A and B to buffer0 and buffer0
      // don't load on last iteration.
      if ( assume(itr < nIt - 1) ){
        int iM = block_start_y + i*32 + dy;
        int iN = block_start_x + i*32 + wx;

        if ( assume(iM < M) ){
          if ( assume(iKA < K) ){
            a_buffer0[i] = load_global(A + (block_id)*M*K + (iM)*K + (iKA));
          } else {
            a_buffer0[i] = 0.f;
          }

          if ( assume(iKA+8 < K) ){
            a_buffer1[i] = load_global(A + (block_id)*M*K + (iM)*K + (iKA+8));
          } else {
            a_buffer1[i] = 0.f;
          }
        }

        if ( assume(iN < N) ){
          if ( assume(iKB < K) ){
            b_buffer0[i] = load_global(B + (block_id)*N*K + (iKB)*N + (iN));
          } else {
            b_buffer0[i] = 0.f;
          }

          if ( assume(iKB+8 < K) ){
            b_buffer1[i] = load_global(B + (block_id)*N*K + (iKB+8)*N + (iN));
          } else {
            b_buffer1[i] = 0.f;
          }
        }
      }
    }
    // sync data loading between threads
    __syncthreads();

    // if thread is in window size compute matmul
    if ( do_thread ){
      matmul_thread(a_shared0, b_shared0, c_cache, vx, vy);
      matmul_thread(a_shared1, b_shared1, c_cache, vx, vy);
    }
    // sync matmul computation between threads
    __syncthreads();
  }

  //write cache to C
  __shared__ volatile float c_shared[16][128];
  #pragma unroll
  for ( int mi = 0; mi < 8; mi++ ){
    // write c_cache (register memory) to c_shared (shared memory)
    int iM = block_start_y + vy*8 + mi;
    if ( iM < M ){
      #pragma unroll
      for ( int ni = 0; ni < 8; ni++ ){
        int iN = block_start_x + vx*8 + ni;
        if ( assume( (iM < iN ) || iM  >= iN +  ws ) ){
          c_shared[vy][vx*8 + ni] = 0.f;
        }else{
          c_shared[vy][vx*8 + ni] = c_cache[mi].val[ni];
        }
      }
      // write c_shared (shared memory) to C (global memory)
      #pragma unroll
      for ( int ni = 0; ni < 8; ni++ ){
        int iN = block_start_x + 16*ni + vx;
        if ( assume( iN < N ) ){
          C[(block_id)*M*N + (iM)*N + (iN)] = c_shared[vy][16*ni + vx];
        }
      }

    }

  }
}


at::Tensor  attention_window_matmul_cuda_forward(
  const at::Tensor a,
  const at::Tensor b,
  const at::Tensor ws,
  const at::Tensor wm,
  const bool masked){


  int bs = a.size(0); //batch size
  int nh = a.size(1); //number of heads
  int m  = a.size(2); //sequence length
  int k  = a.size(3); //head dimension
  int n  = b.size(3); //sequence length

  assert(ws.size(0) == nh);

  at::Tensor c = at::zeros({bs, nh, m, n}, at::TensorOptions().dtype(a.dtype()).requires_grad(true).device(a.device()));

  const dim3 threads ( 256 );
  const dim3 blocks ( 16, 4, bs*nh );

  if (masked){
    masked_kernel<<<blocks, threads>>>(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      c.data_ptr<float>(),
      ws.data_ptr<float>(),
      m, n, k, bs, nh);
  }else{
    not_masked_kernel<<<blocks, threads>>>(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      c.data_ptr<float>(),
      ws.data_ptr<float>(),
      m, n, k, bs, nh);
  }

  return c;
}
