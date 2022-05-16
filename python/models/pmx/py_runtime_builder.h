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

#ifndef _ST_HPC_PPL_NN_PYTHON_PMX_PY_RUNTIME_BUILDER_H_
#define _ST_HPC_PPL_NN_PYTHON_PMX_PY_RUNTIME_BUILDER_H_

#include "ppl/nn/models/pmx/runtime_builder.h"
#include "ppl/nn/engines/engine.h"
#include <vector>
#include <memory>

namespace ppl { namespace nn { namespace python { namespace pmx {

struct PyRuntimeBuilder final {
    PyRuntimeBuilder(ppl::nn::pmx::RuntimeBuilder* b) : ptr(b) {}
    PyRuntimeBuilder(PyRuntimeBuilder&&) = default;
    PyRuntimeBuilder& operator=(PyRuntimeBuilder&&) = default;
    ~PyRuntimeBuilder() {
        ptr.reset();
        engines.clear();
    }
    std::unique_ptr<ppl::nn::pmx::RuntimeBuilder> ptr;
    std::vector<std::shared_ptr<Engine>> engines; // retain engines
};

}}}} // namespace ppl::nn::python::pmx

#endif
