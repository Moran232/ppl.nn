// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "cudakernel/arithmetic/einsum.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_fp16.h>

template <typename T>
__global__ void ppl_cukernel_einsum_nbdce(const T* input0, const T* input1, T* output, uint64_t outer, uint64_t inner,
                                        uint64_t n, uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e){
        // nbac * ndae --> nbdce
        int tid = threadIdx.x;
        int outer_id = blockIdx.x; //nbd
        int inner_id = blockIdx.y; //ce
        int productDim = a;

        int nb_id = outer_id / d;
        int d_id = outer_id % d;

        int n_id = nb_id / b;
        int b_id = nb_id % b;  // outer_id = n_id*b*d + b_id*d + d_id

        int c_id = inner_id / e;
        int e_id = inner_id % e;

        __shared__ T tmp[256];
        tmp[tid] = 0;
        for(int id = tid; id < productDim; id += blockDim.x){
            if(id < productDim){
                uint64_t input0_offset = n_id * b * a * c + b_id * a * c + id * c + c_id;
                uint64_t input1_offset = n_id * d * a * e + d_id * a * e + id * e + e_id;

                tmp[id] += input0[input0_offset] * input1[input1_offset];
            }
        }
        __syncthreads();

        for(int stride=blockDim.x/2; stride>0; stride >>= 1){
            if(tid < stride)
                tmp[tid] += tmp[tid + stride];
            __syncthreads();
        }

        uint64_t output_offset = outer_id * inner + inner_id;
        output[output_offset] = tmp[0];
}
template <typename T>
__global__ void ppl_cukernel_einsum_nbdc(const T* input0, const T* input1, T* output, uint64_t outer, uint64_t inner,
                                        uint64_t n, uint64_t a, uint64_t b, uint64_t c, uint64_t d){
        // nabc * nadc -> n(a)bdc
        int tid = threadIdx.x;
        int outer_id = blockIdx.x; //nc
        int inner_id = blockIdx.y; //bd
        int productDim = a;

        int c_id = outer_id % c;
        int n_id = outer_id / c;

        // inner_id = b_id *d + d_id
        int d_id = inner_id % d;
        int b_id = inner_id / d;

        __shared__ T tmp[256];
        tmp[tid] = 0;
        for(int id = tid; id < productDim; id += blockDim.x){
            if(id < productDim){
                uint64_t input0_offset = n_id * a * b * c + id * b * c + b_id * c + c_id;
                uint64_t input1_offset = n_id * a * d * c + id * d * c + d_id * c + c_id;

                tmp[id] += input0[input0_offset] * input1[input1_offset];
            }
        }
        __syncthreads();

        for(int stride=blockDim.x/2; stride>0; stride >>= 1){
            if(tid < stride)
                tmp[tid] += tmp[tid + stride];
            __syncthreads();
        }
        // ncbd = n_id*c*b*d + c_id*b*d + b_id*d + d_id = outer_id * inner + inner_id
        // nbdc
        uint64_t output_offset = n_id*b*d*c + b_id*d*c + d_id*c + c_id;
        output[output_offset] = tmp[0];
}

ppl::common::RetCode PPLCUDAEinSum_nbac_ndae_nbdce_ForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape0,
    const void* input0,
    const ppl::nn::TensorShape* input_shape1,
    const void* input1,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    std::string equation)
{
    // nbac * ndae -> nbd(a)ce
    auto n = input_shape0->GetDim(0);
    auto b = input_shape0->GetDim(1);
    auto a = input_shape0->GetDim(2);
    auto c = input_shape0->GetDim(3);
    auto d = input_shape1->GetDim(1);
    auto e = input_shape1->GetDim(3);

    auto outer = n * b * d; // nbd
    auto inner = c * e; //ce

    int block_size     = 256;
    dim3 block(block_size);
    dim3 grid(outer, inner);

    auto datatype = output_shape->GetDataType();
    auto dataformat = output_shape->GetDataFormat();

    switch(datatype){
        case ppl::common::DATATYPE_FLOAT32:{
            ppl_cukernel_einsum_nbdce<float><<<grid, block, 0, stream>>>((const float*)input0, (const float*)input1, (float*)output, outer, inner, n, a, b, c, d, e);
            break;
        }
        case ppl::common::DATATYPE_FLOAT16:{
            ppl_cukernel_einsum_nbdce<half><<<grid, block, 0, stream>>>((const half*)input0, (const half*)input1, (half*)output, outer, inner, n, a, b, c, d, e);
            break;
        }
        case ppl::common::DATATYPE_INT64:{
            ppl_cukernel_einsum_nbdce<int64_t><<<grid, block, 0, stream>>>((const int64_t*)input0, (const int64_t*)input1, (int64_t*)output, outer, inner, n, a, b, c, d, e);
            break;
        }
        default:
            return ppl::common::RC_UNSUPPORTED;
    }


    return ppl::common::RC_SUCCESS;
}


ppl::common::RetCode PPLCUDAEinSum_nabc_nadc_nbdc_ForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape0,
    const void* input0,
    const ppl::nn::TensorShape* input_shape1,
    const void* input1,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    std::string equation)
{
    // nabc * nadc -> n(a)bdc
    auto n = input_shape0->GetDim(0);
    auto a = input_shape0->GetDim(1);
    auto b = input_shape0->GetDim(2);
    auto c = input_shape0->GetDim(3);
    auto d = input_shape1->GetDim(2);

    auto outer = n * c; // nc
    auto inner = b * d; // bd

    int block_size     = 256;
    dim3 block(block_size);
    dim3 grid(outer, inner);

    auto datatype = output_shape->GetDataType();
    auto dataformat = output_shape->GetDataFormat();

    switch(datatype){
        case ppl::common::DATATYPE_FLOAT32:{
            ppl_cukernel_einsum_nbdc<float><<<grid, block, 0, stream>>>((const float*)input0, (const float*)input1, (float*)output, outer, inner, n, a, b, c, d);
            break;
        }
        case ppl::common::DATATYPE_FLOAT16:{
            ppl_cukernel_einsum_nbdc<half><<<grid, block, 0, stream>>>((const half*)input0, (const half*)input1, (half*)output, outer, inner, n, a, b, c, d);
            break;
        }
        case ppl::common::DATATYPE_INT64:{
            ppl_cukernel_einsum_nbdc<int64_t><<<grid, block, 0, stream>>>((const int64_t*)input0, (const int64_t*)input1, (int64_t*)output, outer, inner, n, a, b, c, d);
            break;
        }
        default:
            return ppl::common::RC_UNSUPPORTED;
    }


    return ppl::common::RC_SUCCESS;
}