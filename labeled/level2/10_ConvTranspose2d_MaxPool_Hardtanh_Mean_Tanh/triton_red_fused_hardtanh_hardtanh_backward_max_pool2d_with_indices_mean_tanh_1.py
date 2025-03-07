# From: 10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused_hardtanh_hardtanh_backward_max_pool2d_with_indices_mean_tanh_1(
    in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    rnumel = 256
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_index
    _temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r1 = r_index % 16
        r2 = (r_index // 16)
        r3 = r_index

        input_val0 = tl.load(in_ptr0 + ((2 * r1) + (64 * r2) + (1024 * x0)), r_mask, eviction_policy='evict_last', other=0.0)
        input_val1 = tl.load(in_ptr0 + (1 + (2 * r1) + (64 * r2) + (1024 * x0)), r_mask, eviction_policy='evict_last', other=0.0)
        input_val7 = tl.load(in_ptr0 + (32 + (2 * r1) + (64 * r2) + (1024 * x0)), r_mask, eviction_policy='evict_last', other=0.0)
        input_val12 = tl.load(in_ptr0 + (33 + (2 * r1) + (64 * r2) + (1024 * x0)), r_mask, eviction_policy='evict_last', other=0.0)

        is_input1_greater = input_val1 > input_val0
        mask_1 = tl.full([1, 1], 1, tl.int8)
        mask_0 = tl.full([1, 1], 0, tl.int8)
        max_index_0_1 = tl.where(is_input1_greater, mask_1, mask_0)

        max_val_0_1 = triton_helpers.maximum(input_val1, input_val0)
        is_input7_greater = input_val7 > max_val_0_1
        mask_2 = tl.full([1, 1], 2, tl.int8)
        max_index_0_1_7 = tl.where(is_input7_greater, mask_2, max_index_0_1)

        max_val_0_1_7 = triton_helpers.maximum(input_val7, max_val_0_1)
        is_input12_greater = input_val12 > max_val_0_1_7
        mask_3 = tl.full([1, 1], 3, tl.int8)
        max_index = tl.where(is_input12_greater, mask_3, max_index_0_1_7)

        max_val = triton_helpers.maximum(input_val12, max_val_0_1_7)

        lower_bound = -1.0
        upper_bound = 1.0
        is_out_of_bounds = (max_val <= lower_bound) | (max_val >= upper_bound)
        clamped_val = triton_helpers.maximum(max_val, lower_bound)
        clamped_val = triton_helpers.minimum(clamped_val, upper_bound)
        clamped_broadcast = tl.broadcast_to(clamped_val, [XBLOCK, RBLOCK])

        _temp_sum = _temp_sum + clamped_broadcast
        _temp_sum = tl.where(r_mask, _temp_sum, _temp_sum)

        tl.store(out_ptr0 + (r3 + (256 * x0)), max_index, r_mask)
        tl.store(out_ptr1 + (r3 + (256 * x0)), is_out_of_bounds, r_mask)

    sum_temp = tl.sum(_temp_sum, 1)[:, None]
    normalization_factor = 256.0
    mean_val = sum_temp / normalization_factor
    tanh_val = tl.extra.cuda.libdevice.tanh(mean_val)

    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tanh_val, None)