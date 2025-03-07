# From: 3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_avg_pool3d_backward_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_1(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, 
    output_ptr0, output_ptr3, kernel_size0, kernel_size1, kernel_size2, 
    x_num_elements, r_num_elements, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = r_index
    x0 = (x_index % 64)
    x1 = ((x_index // 64) % kernel_size0)
    x2 = x_index // kernel_size1
    x4 = x_index

    tmp0 = tl.load(
        input_ptr0 + (
            32 * (
                ((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))
                * (((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0))) <= ((-1) + ((32) * ((32) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (32)))))
                + ((-1) + ((32) * ((32) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (32))))
                * (((-1) + ((32) * ((32) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (32)))) < (((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))))
            )
            + 1024 * (
                ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))
                * (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0))) <= ((-1) + ((kernel_size2) * ((kernel_size2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (kernel_size2)))))
                + ((-1) + ((kernel_size2) * ((kernel_size2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (kernel_size2))))
                * (((-1) + ((kernel_size2) * ((kernel_size2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (kernel_size2)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))
            )
            + 1024 * kernel_size2 * x2
            + (
                ((0) * ((0) >= (r3 // 2)) + (r3 // 2) * ((r3 // 2) > (0)))
                * (((0) * ((0) >= (r3 // 2)) + (r3 // 2) * ((r3 // 2) > (0))) <= ((-1) + ((32) * ((32) <= (1 + (r3 // 2))) + (1 + (r3 // 2)) * ((1 + (r3 // 2)) < (32)))))
                + ((-1) + ((32) * ((32) <= (1 + (r3 // 2))) + (1 + (r3 // 2)) * ((1 + (r3 // 2)) < (32))))
                * (((-1) + ((32) * ((32) <= (1 + (r3 // 2))) + (1 + (r3 // 2)) * ((1 + (r3 // 2)) < (32)))) < (((0) * ((0) >= (r3 // 2)) + (r3 // 2) * ((r3 // 2) > (0)))))
            )
        ), 
        None, 
        eviction_policy='evict_last'
    )

    tmp15 = tl.load(input_ptr1 + (r3), None, eviction_policy='evict_last')
    tmp20 = tl.load(input_ptr2 + (r3 + 64 * x4), None)
    tmp21 = tl.load(input_ptr3 + (0))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.load(input_ptr4 + (x4), None, eviction_policy='evict_last')
    tmp26 = tl.load(input_ptr5 + (x4), None, eviction_policy='evict_last')

    tmp1 = tmp0 / 8
    tmp2 = ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))
    tmp3 = ((kernel_size2) * ((kernel_size2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (kernel_size2)))
    tmp4 = tmp2 < tmp3
    tmp5 = ((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))
    tmp6 = ((32) * ((32) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (32)))
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = ((0) * ((0) >= (r3 // 2)) + (r3 // 2) * ((r3 // 2) > (0)))
    tmp10 = ((32) * ((32) <= (1 + (r3 // 2))) + (1 + (r3 // 2)) * ((1 + (r3 // 2)) < (32)))
    tmp11 = tmp9 < tmp10
    tmp12 = tmp8 & tmp11
    tmp13 = 0.0
    tmp14 = tl.where(tmp12, tmp1, tmp13)

    tmp16 = tmp14 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.sum(tmp17, 1)[:, None]

    tmp23 = tmp20 + tmp22
    tmp25 = tmp23 - tmp24
    tmp27 = tmp25 * tmp26
    tmp28 = tmp16 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.sum(tmp29, 1)[:, None]

    tmp32 = 0.015625
    tmp33 = tmp26 * tmp32
    tmp34 = 64.0
    tmp35 = tmp16 * tmp34
    tmp36 = tmp35 - tmp19
    tmp37 = tmp27 * tmp31
    tmp38 = tmp36 - tmp37
    tmp39 = tmp33 * tmp38

    tl.store(output_ptr0 + (r3 + 64 * x4), tmp14, None)
    tl.store(output_ptr3 + (r3 + 64 * x4), tmp39, None)