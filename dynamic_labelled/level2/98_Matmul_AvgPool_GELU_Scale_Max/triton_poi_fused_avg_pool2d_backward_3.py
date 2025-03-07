# From: 98_Matmul_AvgPool_GELU_Scale_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool2d_backward_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    program_id = tl.program_id(0)
    xoffset = program_id * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel

    x_mod_256 = xindex % 256
    x_div_256 = xindex // 256
    x_full_index = xindex

    load_offset = 64 * x_div_256 + (
        (
            (0) * ((0) >= (x_mod_256 // 4)) + (x_mod_256 // 4) * ((x_mod_256 // 4) > (0))
        ) * (
            (
                (0) * ((0) >= (x_mod_256 // 4)) + (x_mod_256 // 4) * ((x_mod_256 // 4) > (0))
            ) <= (
                (-1) + (
                    (64) * ((64) <= (1 + (x_mod_256 // 4))) + (1 + (x_mod_256 // 4)) * ((1 + (x_mod_256 // 4)) < (64))
                )
            )
        ) + (
            (-1) + (
                (64) * ((64) <= (1 + (x_mod_256 // 4))) + (1 + (x_mod_256 // 4)) * ((1 + (x_mod_256 // 4)) < (64))
            )
        ) * (
            (
                (-1) + (
                    (64) * ((64) <= (1 + (x_mod_256 // 4))) + (1 + (x_mod_256 // 4)) * ((1 + (x_mod_256 // 4)) < (64))
                )
            ) < (
                (0) * ((0) >= (x_mod_256 // 4)) + (x_mod_256 // 4) * ((x_mod_256 // 4) > (0))
            )
        )
    )

    loaded_data = tl.load(in_ptr0 + load_offset, xmask, eviction_policy='evict_last')
    averaged_data = loaded_data / 4

    zero_tensor = tl.full([1], 0, tl.int32)
    one_tensor = tl.full([1], 1, tl.int32)
    zero_less_one = zero_tensor < one_tensor

    condition1 = (0) * ((0) >= (x_mod_256 // 4)) + (x_mod_256 // 4) * ((x_mod_256 // 4) > (0))
    condition2 = (64) * ((64) <= (1 + (x_mod_256 // 4))) + (1 + (x_mod_256 // 4)) * ((1 + (x_mod_256 // 4)) < (64))
    condition3 = condition1 < condition2

    combined_condition = zero_less_one & condition3

    zero_float = 0.0
    result_data = tl.where(combined_condition, averaged_data, zero_float)

    tl.store(out_ptr0 + x_full_index, result_data, xmask)