# From: 60_ConvTranspose3d_Swish_GroupNorm_HardSwish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_0(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, 
    output_ptr0, kernel_size0, kernel_size1, kernel_size2, kernel_size3, 
    kernel_size4, kernel_size5, kernel_size6, num_elements, block_size: tl.constexpr
):
    offset = tl.program_id(0) * block_size
    index = offset + tl.arange(0, block_size)[:]
    mask = index < num_elements

    x0 = (index % kernel_size0)
    x1 = ((index // kernel_size0) % kernel_size0)
    x2 = ((index // kernel_size1) % kernel_size2)
    x5 = index // kernel_size3
    x8 = index // kernel_size6
    x3 = ((index // kernel_size3) % 16)
    x9 = index

    tmp0 = tl.load(
        input_ptr0 + (
            x0 + 
            (-1) * x5 + 
            (-1) * (
                (
                    (x0 + x2 + (-1) * x1 + (-4) * kernel_size5 * x2 + 
                     2 * kernel_size5 * x1 + 4 * x2 * kernel_size5 * kernel_size5) // 
                    (-1 + 2 * kernel_size5)
                ) % 
                (-1 + 2 * kernel_size5)
            ) + 
            (-4) * kernel_size5 * (
                (
                    (x0 + x2 + (-1) * x1 + (-4) * kernel_size5 * x2 + 
                     2 * kernel_size5 * x1 + 4 * x2 * kernel_size5 * kernel_size5) // 
                    (1 + (-4) * kernel_size5 + 4 * kernel_size5 * kernel_size5)
                ) % 
                (-1 + 2 * kernel_size4)
            ) + 
            (-4) * x5 * kernel_size5 * kernel_size5 + 
            2 * kernel_size4 * x5 + 
            2 * kernel_size5 * (
                (
                    (x0 + x2 + (-1) * x1 + (-4) * kernel_size5 * x2 + 
                     2 * kernel_size5 * x1 + 4 * x2 * kernel_size5 * kernel_size5) // 
                    (-1 + 2 * kernel_size5)
                ) % 
                (-1 + 2 * kernel_size5)
            ) + 
            4 * kernel_size5 * x5 + 
            4 * kernel_size5 * kernel_size5 * (
                (
                    (x0 + x2 + (-1) * x1 + (-4) * kernel_size5 * x2 + 
                     2 * kernel_size5 * x1 + 4 * x2 * kernel_size5 * kernel_size5) // 
                    (1 + (-4) * kernel_size5 + 4 * kernel_size5 * kernel_size5)
                ) % 
                (-1 + 2 * kernel_size4)
            ) + 
            (-8) * kernel_size4 * kernel_size5 * x5 + 
            8 * kernel_size4 * x5 * kernel_size5 * kernel_size5 + 
            (
                (
                    (x0 + x2 + (-1) * x1 + (-4) * kernel_size5 * x2 + 
                     2 * kernel_size5 * x1 + 4 * x2 * kernel_size5 * kernel_size5) // 
                    (1 + (-4) * kernel_size5 + 4 * kernel_size5 * kernel_size5)
                ) % 
                (-1 + 2 * kernel_size4)
            )
        ), 
        mask, 
        eviction_policy='evict_last'
    )

    tmp2 = tl.load(
        input_ptr0 + (
            x0 + 
            (-1) * x5 + 
            (-1) * (
                (
                    (x0 + x2 + (-1) * x1 + (-4) * kernel_size5 * x2 + 
                     2 * kernel_size5 * x1 + 4 * x2 * kernel_size5 * kernel_size5) // 
                    kernel_size0
                ) % 
                kernel_size0
            ) + 
            (-4) * kernel_size5 * (
                (
                    (x0 + x2 + (-1) * x1 + (-4) * kernel_size5 * x2 + 
                     2 * kernel_size5 * x1 + 4 * x2 * kernel_size5 * kernel_size5) // 
                    (1 + (-4) * kernel_size5 + 4 * kernel_size5 * kernel_size5)
                ) % 
                kernel_size2
            ) + 
            (-4) * x5 * kernel_size5 * kernel_size5 + 
            2 * kernel_size4 * x5 + 
            2 * kernel_size5 * (
                (
                    (x0 + x2 + (-1) * x1 + (-4) * kernel_size5 * x2 + 
                     2 * kernel_size5 * x1 + 4 * x2 * kernel_size5 * kernel_size5) // 
                    kernel_size0
                ) % 
                kernel_size0
            ) + 
            4 * kernel_size5 * x5 + 
            4 * kernel_size5 * kernel_size5 * (
                (
                    (x0 + x2 + (-1) * x1 + (-4) * kernel_size5 * x2 + 
                     2 * kernel_size5 * x1 + 4 * x2 * kernel_size5 * kernel_size5) // 
                    (1 + (-4) * kernel_size5 + 4 * kernel_size5 * kernel_size5)
                ) % 
                kernel_size2
            ) + 
            (-8) * kernel_size4 * kernel_size5 * x5 + 
            8 * kernel_size4 * x5 * kernel_size5 * kernel_size5 + 
            (
                (
                    (x0 + x2 + (-1) * x1 + (-4) * kernel_size5 * x2 + 
                     2 * kernel_size5 * x1 + 4 * x2 * kernel_size5 * kernel_size5) // 
                    (1 + (-4) * kernel_size5 + 4 * kernel_size5 * kernel_size5)
                ) % 
                kernel_size2
            )
        ), 
        mask, 
        eviction_policy='evict_last'
    )

    tmp4 = tl.load(input_ptr1 + (x8 // 4), mask, eviction_policy='evict_last')
    tmp6 = tl.load(input_ptr2 + (x8 // 4), mask, eviction_policy='evict_last')
    tmp8 = tl.load(input_ptr3 + (x3), mask, eviction_policy='evict_last')
    tmp10 = tl.load(input_ptr4 + (x3), mask, eviction_policy='evict_last')

    tmp1 = tl.sigmoid(tmp0)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 - tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 + tmp10

    tl.store(output_ptr0 + (x9), tmp11, mask)