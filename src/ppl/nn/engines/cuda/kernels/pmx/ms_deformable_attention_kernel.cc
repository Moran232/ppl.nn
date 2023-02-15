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

#include "ppl/nn/engines/cuda/kernels/pmx/ms_deformable_attention_kernel.h"
#include "ppl/common/destructor.h"

#include <numeric>

#include "cudakernel/nn/ms_deformable_attention.h"

namespace ppl { namespace nn { namespace cuda {

/* uint64_t MsDeformAttnKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const { */
    /* auto y = ctx.GetOutput<TensorImpl>(0); */
    /* if (y->GetShape()->GetDataType() == ppl::common::DATATYPE_INT8) { */
        /* return sizeof(float) * y->GetShape()->CalcElementsExcludingPadding(); */
    /* } else { */
        /* return 0; */
    /* } */
/* } */

ppl::common::RetCode MsDeformAttnKernel::DoExecute(KernelExecContext* ctx) {

    auto input0 = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    // const TensorShape& input_shape = *input0->GetShape();

    // auto input_quant = GetCommonParam()->cuda_tensor_info->at(input->GetEdge()->GetId());
    // auto output_quant = GetCommonParam()->cuda_tensor_info->at(output->GetEdge()->GetId());
    // QuantKernelParamCuda qparam(input_quant.zero_point[0], output_quant.zero_point[0], input_quant.scale[0], output_quant.scale[0]);

    auto input1 = ctx->GetInput<TensorImpl>(1);
    auto input2 = ctx->GetInput<TensorImpl>(2);
    auto input3 = ctx->GetInput<TensorImpl>(3);
    auto input4 = ctx->GetInput<TensorImpl>(4);

    auto status =
        PPLCUDAMsDeformAttnForwardImp(GetStream(),
                input0->GetBufferPtr(),
                input1->GetBufferPtr(),
                input2->GetBufferPtr(),
                input3->GetBufferPtr(),
                input4->GetBufferPtr(),
                output->GetBufferPtr(), param_->im2col_step);
    return status;

}

}}} // namespace ppl::nn::cuda
