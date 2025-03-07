# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_backward_0(input_grad_ptr, input_ptr, output_grad_ptr, output_ptr, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 1024
    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    r_block_mask = tl.full([RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[:]
    r1 = r_index
    x0 = x_index
    input_grad_value = tl.load(input_grad_ptr + (r1 + 1024 * x0), None)
    input_value = tl.load(input_ptr + (r1 + 1024 * x0), None)
    product = input_grad_value * input_value
    broadcast_product = tl.broadcast_to(product, [RBLOCK])
    sum_product = triton_helpers.promote_to_tensor(tl.sum(broadcast_product, 0))
    broadcast_input = tl.broadcast_to(input_value, [RBLOCK])
    sum_input = triton_helpers.promote_to_tensor(tl.sum(broadcast_input, 0))
    tl.store(output_grad_ptr + (x0), sum_product, None)
    tl.store(output_ptr + (x0), sum_input, None)