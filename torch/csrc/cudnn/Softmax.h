#pragma once

#include "../Types.h"
#include <cudnn.h>
#include "THC/THC.h"


namespace torch { namespace cudnn {

void cudnn_softmax_forward(
    THCState* state, cudnnHandle_t handle,
    const at::Tensor& input, at::Tensor& output,
    bool log_softmax);

void cudnn_softmax_backward(
    THCState* state, cudnnHandle_t handle,
    const at::Tensor& output, const at::Tensor& grad_output, at::Tensor& grad_input,
    bool log_softmax);

}}
