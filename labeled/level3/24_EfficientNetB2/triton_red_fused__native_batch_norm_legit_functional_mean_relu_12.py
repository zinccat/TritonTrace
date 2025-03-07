# From: 24_EfficientNetB2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_mean_relu_12red_fused__native_batch_norm_legit_functional_mean_relu_12(
    in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 192
    rnumel = 98
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_channel = (x_index % 96)
    x_batch = x_index // 96
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x_flat_index = x_index

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r_channel = r_index
        loaded_values = tl.load(
            in_ptr0 + (x_channel + 96 * r_channel + 9408 * x_batch),
            r_mask & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        temp_sum += broadcasted_values
        temp_sum = tl.where(r_mask & x_mask, temp_sum, temp_sum)

    sum_over_blocks = tl.sum(temp_sum, 1)[:, None]
    normalization_factor = 12544.0
    normalized_values = sum_over_blocks / normalization_factor

    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x_flat_index), normalized_values, x_mask)