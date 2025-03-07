# From: 21_Conv2d_Add_Scale_Sigmoid_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_2poi_fused_native_group_norm_2(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, 
    output_ptr0, kernel_size0, kernel_size1, kernel_size2, kernel_size3, num_elements, 
    XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < num_elements

    x0 = (x_index % kernel_size0)
    x1 = ((x_index // kernel_size0) % kernel_size0)
    x4 = x_index // kernel_size1
    x2 = ((x_index // kernel_size1) % 16)
    x7 = x_index // kernel_size3
    x8 = x_index

    input0_index = (
        x0 + 
        ((-2) * (((x0 + ((-2) * x1) + kernel_size2 * x1) // ((-2) + kernel_size2)) % ((-2) + kernel_size2))) + 
        4 * x4 + 
        kernel_size2 * ((((x0 + ((-2) * x1) + kernel_size2 * x1) // ((-2) + kernel_size2)) % ((-2) + kernel_size2))) + 
        x4 * kernel_size2 * kernel_size2 + 
        ((-4) * kernel_size2 * x4)
    )
    tmp0 = tl.load(input_ptr0 + input0_index, x_mask, eviction_policy='evict_last')

    tmp1 = tl.load(input_ptr1 + (x2), x_mask, eviction_policy='evict_last')
    tmp3 = tl.load(input_ptr2 + (x2), x_mask, eviction_policy='evict_last')
    tmp6 = tl.load(input_ptr3 + (x7 // 2), x_mask, eviction_policy='evict_last')
    tmp8 = tl.load(input_ptr4 + (x7 // 2), x_mask, eviction_policy='evict_last')
    tmp16 = tl.load(input_ptr5 + (x2), x_mask, eviction_policy='evict_last')
    tmp18 = tl.load(input_ptr6 + (x2), x_mask, eviction_policy='evict_last')

    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.sigmoid(tmp4)
    tmp7 = tmp5 - tmp6

    epsilon = 8 + ((-8) * kernel_size2) + 2 * kernel_size2 * kernel_size2
    tmp9 = (
        tl.full([], 0.0, tl.float64) * (tl.full([], 0.0, tl.float64) >= epsilon) + 
        epsilon * (epsilon > tl.full([], 0.0, tl.float64))
    )
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 / tmp10

    epsilon_value = 1e-05
    tmp12 = tmp11 + epsilon_value
    tmp13 = tl.extra.cuda.libdevice.rsqrt(tmp12)
    tmp14 = tmp7 * tmp13
    tmp15 = tmp14 * tmp16
    tmp17 = tmp15 + tmp18

    tl.store(output_ptr0 + (x8), tmp17, x_mask)