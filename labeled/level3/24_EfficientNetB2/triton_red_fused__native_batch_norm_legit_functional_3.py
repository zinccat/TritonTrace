# From: 24_EfficientNetB2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_3red_fused__native_batch_norm_legit_functional_3(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, output_ptr1, output_ptr2, 
    num_elements_x, num_elements_r, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements_x = 64
    num_elements_r = 98
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_channel = (x_indices % 32)
    x_batch = x_indices // 32
    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_flat_index = x_indices

    for r_offset in range(0, num_elements_r, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_elements_r
        r_channel = r_indices
        input0 = tl.load(input_ptr0 + (x_channel + 32 * r_channel + 3136 * x_batch), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        input1 = tl.load(input_ptr1 + (x_channel + 32 * r_channel + 3136 * x_batch), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        input2 = tl.load(input_ptr2 + (x_channel + 32 * r_channel + 3136 * x_batch), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        
        broadcast_input0 = tl.broadcast_to(input0, [XBLOCK, RBLOCK])
        broadcast_input1 = tl.broadcast_to(input1, [XBLOCK, RBLOCK])
        broadcast_input2 = tl.broadcast_to(input2, [XBLOCK, RBLOCK])
        
        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_combine(
            running_mean, running_m2, running_weight,
            broadcast_input0, broadcast_input1, broadcast_input2
        )
        
        running_mean = tl.where(r_mask & x_mask, running_mean_next, running_mean)
        running_m2 = tl.where(r_mask & x_mask, running_m2_next, running_m2)
        running_weight = tl.where(r_mask & x_mask, running_weight_next, running_weight)

    mean, variance, weight = triton_helpers.welford(running_mean, running_m2, running_weight, 1)
    mean = mean[:, None]
    variance = variance[:, None]
    weight = weight[:, None]

    tl.store(output_ptr0 + (x_flat_index), mean, x_mask)
    tl.store(output_ptr1 + (x_flat_index), variance, x_mask)
    tl.store(output_ptr2 + (x_flat_index), weight, x_mask)