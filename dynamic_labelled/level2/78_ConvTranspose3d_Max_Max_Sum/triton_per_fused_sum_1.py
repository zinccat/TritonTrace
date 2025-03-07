# From: 78_ConvTranspose3d_Max_Max_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_sum_1per_fused_sum_1(in_ptr0, out_ptr0, kernel_size_z, kernel_size_y, kernel_size_x, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 16
    program_id_offset = tl.program_id(0) * XBLOCK
    input_indices = program_id_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < input_num_elements
    reduction_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    
    reduction_index = reduction_indices
    input_mod_z = input_indices % kernel_size_z
    input_div_z = input_indices // kernel_size_z
    input_linear_index = input_indices
    
    tmp0 = tl.load(
        in_ptr0 + (
            reduction_index + input_mod_z + 16 * input_div_z +
            reduction_index * (triton_helpers.div_floor_integer((-4) + kernel_size_z, 3))**2 +
            reduction_index * (triton_helpers.div_floor_integer((-4) + kernel_size_y, 3)) +
            2 * reduction_index * (triton_helpers.div_floor_integer((-4) + kernel_size_z, 3)) +
            16 * input_div_z * (triton_helpers.div_floor_integer((-4) + kernel_size_z, 3))**2 +
            16 * input_div_z * (triton_helpers.div_floor_integer((-4) + kernel_size_y, 3)) +
            32 * input_div_z * (triton_helpers.div_floor_integer((-4) + kernel_size_z, 3)) +
            reduction_index * (triton_helpers.div_floor_integer((-4) + kernel_size_z, 3))**2 * (triton_helpers.div_floor_integer((-4) + kernel_size_y, 3)) +
            2 * reduction_index * (triton_helpers.div_floor_integer((-4) + kernel_size_y, 3)) * (triton_helpers.div_floor_integer((-4) + kernel_size_z, 3)) +
            16 * input_div_z * (triton_helpers.div_floor_integer((-4) + kernel_size_z, 3))**2 * (triton_helpers.div_floor_integer((-4) + kernel_size_y, 3)) +
            32 * input_div_z * (triton_helpers.div_floor_integer((-4) + kernel_size_y, 3)) * (triton_helpers.div_floor_integer((-4) + kernel_size_z, 3))
        ),
        input_mask,
        eviction_policy='evict_last',
        other=0.0
    )
    
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(input_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (input_linear_index), tmp4, input_mask)