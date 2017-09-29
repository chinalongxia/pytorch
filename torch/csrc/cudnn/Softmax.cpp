#include "Conv.h"

#include "THC/THC.h"
#include "Exceptions.h"
#include "Types.h"

#include <cudnn.h>
#include <cstdint>

namespace torch { namespace cudnn {

void cudnn_softmax_forward(
    THCState* state, cudnnHandle_t handle,
    const at::Tensor& input, at::Tensor& output,
    bool log_softmax) {
  assertSameGPU(input, output);
  auto dataType = getDataType(input);

  output.resize_(input.sizes());
  TensorDescriptor idesc(input, 4);
  TensorDescriptor odesc(output, 4);
  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  CHECK(cudnnSoftmaxForward(handle,
                            log_softmax ? CUDNN_SOFTMAX_LOG : CUDNN_SOFTMAX_ACCURATE,
                            CUDNN_SOFTMAX_MODE_CHANNEL,
                            &one,
                            idesc.desc, input.data_ptr(),
                            &zero,
                            odesc.desc, output.data_ptr()));
}

void cudnn_softmax_backward(
    THCState* state, cudnnHandle_t handle,
    const at::Tensor& output, const at::Tensor& grad_output, at::Tensor& grad_input,
    bool log_softmax) {
  assertSameGPU(output, grad_output, grad_input);
  auto dataType = getDataType(output);

  TensorDescriptor odesc(output, 4);
  TensorDescriptor godesc(grad_output, 4);
  TensorDescriptor gidesc(grad_input, 4);
  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  CHECK(cudnnSoftmaxBackward(handle,
                             log_softmax ? CUDNN_SOFTMAX_LOG : CUDNN_SOFTMAX_ACCURATE,
                             CUDNN_SOFTMAX_MODE_CHANNEL,
                             &one,
                             odesc.desc, output.data_ptr(),
                             godesc.desc, grad_output.data_ptr(),
                             &zero,
                             gidesc.desc, grad_input.data_ptr()));
}


}}
