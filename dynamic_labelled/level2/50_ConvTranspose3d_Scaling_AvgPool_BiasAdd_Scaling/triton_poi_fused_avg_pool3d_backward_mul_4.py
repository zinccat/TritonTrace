# From: 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool3d_backward_mul_4(
    input_ptr0, input_ptr1, output_ptr0, kernel_size0, kernel_size1, kernel_size2, kernel_size3, 
    kernel_size4, kernel_size5, kernel_size6, num_elements, BLOCK_SIZE: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK_SIZE
    index = offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = index < num_elements

    x = index % kernel_size0
    y = (index // kernel_size0) % kernel_size0
    z = (index // kernel_size1) % kernel_size2
    d = index // kernel_size3
    z6 = (index // kernel_size6) % kernel_size2
    linear_index = index

    tmp0 = tl.load(
        input_ptr0 + (
            (-1) * d + 
            (-1) * (
                (0 if y // 2 == 0 else y // 2) * 
                ((0 if y // 2 == 0 else y // 2) <= (-1 + ((-1 + kernel_size5) * ((-1 + kernel_size5) <= (1 + (y // 2))) + (1 + (y // 2)) * ((1 + (y // 2)) < (-1 + kernel_size5))))) + 
                (-1 + ((-1 + kernel_size5) * ((-1 + kernel_size5) <= (1 + (y // 2))) + (1 + (y // 2)) * ((1 + (y // 2)) < (-1 + kernel_size5)))) * 
                ((-1 + ((-1 + kernel_size5) * ((-1 + kernel_size5) <= (1 + (y // 2))) + (1 + (y // 2)) * ((1 + (y // 2)) < (-1 + kernel_size5)))) < (0 if y // 2 == 0 else y // 2))
            ) * 
            kernel_size4 + 
            kernel_size5 * (
                (0 if y // 2 == 0 else y // 2) * 
                ((0 if y // 2 == 0 else y // 2) <= (-1 + ((-1 + kernel_size5) * ((-1 + kernel_size5) <= (1 + (y // 2))) + (1 + (y // 2)) * ((1 + (y // 2)) < (-1 + kernel_size5))))) + 
                (-1 + ((-1 + kernel_size5) * ((-1 + kernel_size5) <= (1 + (y // 2))) + (1 + (y // 2)) * ((1 + (y // 2)) < (-1 + kernel_size5)))) * 
                ((-1 + ((-1 + kernel_size5) * ((-1 + kernel_size5) <= (1 + (y // 2))) + (1 + (y // 2)) * ((1 + (y // 2)) < (-1 + kernel_size5)))) < (0 if y // 2 == 0 else y // 2))
            ) + 
            kernel_size5 * kernel_size5 * (
                (0 if z // 2 == 0 else z // 2) * 
                ((0 if z // 2 == 0 else z // 2) <= (-1 + ((-1 + kernel_size4) * ((-1 + kernel_size4) <= (1 + (z // 2))) + (1 + (z // 2)) * ((1 + (z // 2)) < (-1 + kernel_size4))))) + 
                (-1 + ((-1 + kernel_size4) * ((-1 + kernel_size4) <= (1 + (z // 2))) + (1 + (z // 2)) * ((1 + (z // 2)) < (-1 + kernel_size4)))) * 
                ((-1 + ((-1 + kernel_size4) * ((-1 + kernel_size4) <= (1 + (z // 2))) + (1 + (z // 2)) * ((1 + (z // 2)) < (-1 + kernel_size4)))) < (0 if z // 2 == 0 else z // 2))
            ) + 
            (-1) * d * kernel_size5 * kernel_size5 + 
            (-2) * kernel_size5 * (
                (0 if z // 2 == 0 else z // 2) * 
                ((0 if z // 2 == 0 else z // 2) <= (-1 + ((-1 + kernel_size4) * ((-1 + kernel_size4) <= (1 + (z // 2))) + (1 + (z // 2)) * ((1 + (z // 2)) < (-1 + kernel_size4))))) + 
                (-1 + ((-1 + kernel_size4) * ((-1 + kernel_size4) <= (1 + (z // 2))) + (1 + (z // 2)) * ((1 + (z // 2)) < (-1 + kernel_size4)))) * 
                ((-1 + ((-1 + kernel_size4) * ((-1 + kernel_size4) <= (1 + (z // 2))) + (1 + (z // 2)) * ((1 + (z // 2)) < (-1 + kernel_size4)))) < (0 if z // 2 == 0 else z // 2))
            ) + 
            2 * kernel_size5 * d + 
            kernel_size4 * d * kernel_size5 * kernel_size5 + 
            (-2) * kernel_size4 * kernel_size5 * d + 
            (0 if x // 2 == 0 else x // 2) * 
            ((0 if x // 2 == 0 else x // 2) <= (-1 + ((-1 + kernel_size5) * ((-1 + kernel_size5) <= (1 + (x // 2))) + (1 + (x // 2)) * ((1 + (x // 2)) < (-1 + kernel_size5))))) + 
            (-1 + ((-1 + kernel_size5) * ((-1 + kernel_size5) <= (1 + (x // 2))) + (1 + (x // 2)) * ((1 + (x // 2)) < (-1 + kernel_size5)))) * 
            ((-1 + ((-1 + kernel_size5) * ((-1 + kernel_size5) <= (1 + (x // 2))) + (1 + (x // 2)) * ((1 + (x // 2)) < (-1 + kernel_size5)))) < (0 if x // 2 == 0 else x // 2)) + 
            (0 if z // 2 == 0 else z // 2) * 
            ((0 if z // 2 == 0 else z // 2) <= (-1 + ((-1 + kernel_size4) * ((-1 + kernel_size4) <= (1 + (z // 2))) + (1 + (z // 2)) * ((1 + (z // 2)) < (-1 + kernel_size4))))) + 
            (-1 + ((-1 + kernel_size4) * ((-1 + kernel_size4) <= (1 + (z // 2))) + (1 + (z // 2)) * ((1 + (z // 2)) < (-1 + kernel_size4)))) * 
            ((-1 + ((-1 + kernel_size4) * ((-1 + kernel_size4) <= (1 + (z // 2))) + (1 + (z // 2)) * ((1 + (z // 2)) < (-1 + kernel_size4)))) < (0 if z // 2 == 0 else z // 2))
        ), 
        mask, eviction_policy='evict_last'
    )

    tmp1 = tl.load(input_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [BLOCK_SIZE])
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3 / 8

    tmp5 = (0 if z6 // 2 == 0 else z6 // 2)
    tmp6 = ((-1 + kernel_size4) * ((-1 + kernel_size4) <= (1 + (z6 // 2))) + (1 + (z6 // 2)) * ((1 + (z6 // 2)) < (-1 + kernel_size4)))
    tmp7 = tmp5 < tmp6

    tmp8 = (0 if y // 2 == 0 else y // 2)
    tmp9 = ((-1 + kernel_size5) * ((-1 + kernel_size5) <= (1 + (y // 2))) + (1 + (y // 2)) * ((1 + (y // 2)) < (-1 + kernel_size5)))
    tmp10 = tmp8 < tmp9

    tmp11 = tmp7 & tmp10

    tmp12 = (0 if x // 2 == 0 else x // 2)
    tmp13 = ((-1 + kernel_size5) * ((-1 + kernel_size5) <= (1 + (x // 2))) + (1 + (x // 2)) * ((1 + (x // 2)) < (-1 + kernel_size5)))
    tmp14 = tmp12 < tmp13

    tmp15 = tmp11 & tmp14

    tmp16 = 0.0
    tmp17 = tl.where(tmp15, tmp4, tmp16)

    tl.store(output_ptr0 + (linear_index), tmp17, mask)