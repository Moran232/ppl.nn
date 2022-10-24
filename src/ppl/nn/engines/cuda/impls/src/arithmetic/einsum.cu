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
#include "cudakernel/memory/transpose.h"
#include "cudakernel/memory/unsqueeze.h"
#include "ppl/common/destructor.h"

#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include "cudakernel/common/common.h"
#include "ppl/nn/common/logger.h"

#include "ppl/nn/oputils/onnx/reshape_einsum.h"

#include "ppl/common/types.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <string>
#include <vector>

ppl::common::RetCode PPLCUDAEinSum2InputForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape0,
    const void* input0,
    const ppl::nn::TensorShape* input_shape1,
    const void* input1,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    std::string equation){


    return  ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDAEinSum1InputForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape0,
    const void* input0,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    std::string equation){
        return 0;

}


