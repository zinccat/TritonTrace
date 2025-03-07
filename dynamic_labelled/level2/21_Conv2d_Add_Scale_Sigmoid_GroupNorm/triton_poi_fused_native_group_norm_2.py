# From: 21_Conv2d_Add_Scale_Sigmoid_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_2(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, 
    output_ptr0, kernel_size0, kernel_size1, kernel_size2, kernel_size3, num_elements, 
    XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    x0 = (index % kernel_size0)
    x1 = ((index // kernel_size0) % kernel_size0)
    x4 = index // kernel_size1
    x2 = ((index // kernel_size1) % 16)
    x7 = index // kernel_size3
    x8 = index

    input_val0 = tl.load(
        input_ptr0 + (x0 + ((-2) * (((x0 + ((-2) * x1) + kernel_size2 * x1) // ((-2) + kernel_size2)) % ((-2) + kernel_size2))) + 4 * x4 + kernel_size2 * ((((x0 + ((-2) * x1) + kernel_size2 * x1) // ((-2) + kernel_size2)) % ((-2) + kernel_size2))) + x4 * kernel_size2 * kernel_size2 + ((-4) * kernel_size2 * x4)), 
        mask, 
        eviction_policy='evict_last'
    )
    input_val1 = tl.load(input_ptr1 + (x2), mask, eviction_policy='evict_last')
    input_val3 = tl.load(input_ptr2 + (x2), mask, eviction_policy='evict_last')
    input_val6 = tl.load(input_ptr3 + (x7 // 2), mask, eviction_policy='evict_last')
    input_val8 = tl.load(input_ptr4 + (x7 // 2), mask, eviction_policy='evict_last')
    input_val16 = tl.load(input_ptr5 + (x2), mask, eviction_policy='evict_last')
    input_val18 = tl.load(input_ptr6 + (x2), mask, eviction_policy='evict_last')

    sum_val = input_val0 + input_val1
    product_val = sum_val * input_val3
    sigmoid_val = tl.sigmoid(product_val)
    diff_val = sigmoid_val - input_val6

    epsilon_val = 8 + ((-8) * kernel_size2) + 2 * kernel_size2 * kernel_size2
    condition_val = (tl.full([], 0.0, tl.float64)) >= epsilon_val
    adjusted_epsilon = (tl.full([], 0.0, tl.float64)) * condition_val + epsilon_val * (epsilon_val > (tl.full([], 0.0, tl.float64)))
    adjusted_epsilon = adjusted_epsilon.to(tl.float32)

    inv_sqrt_val = input_val8 / adjusted_epsilon
    epsilon_small = 1e-05
    inv_sqrt_val += epsilon_small
    rsqrt_val = tl.extra.cuda.libdevice.rsqrt(inv_sqrt_val)

    scaled_diff = diff_val * rsqrt_val
    final_product = scaled_diff * input_val16
    output_val = final_product + input_val18

    tl.store(output_ptr0 + (x8), output_val, mask)