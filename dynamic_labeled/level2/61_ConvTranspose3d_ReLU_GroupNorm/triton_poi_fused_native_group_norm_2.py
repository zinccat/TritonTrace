# From: 61_ConvTranspose3d_ReLU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_2(
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
    x3 = ((index // kernel_size3) % 128)
    x9 = index

    tmp0 = tl.load(
        input_ptr0 + (
            x0 + 2 * (((x1 + 2 * x2 + kernel_size4 * x2) % (2 + kernel_size4))) +
            4 * (((((x0 + 2 * x1 + 4 * x2 + kernel_size4 * x1 + x2 * kernel_size4 * kernel_size4 + 4 * kernel_size4 * x2) // (4 + kernel_size4 * kernel_size4 + 4 * kernel_size4)) % (2 + kernel_size5)))) +
            8 * x5 + kernel_size4 * (((x1 + 2 * x2 + kernel_size4 * x2) % (2 + kernel_size4))) +
            kernel_size4 * kernel_size4 * (((((x0 + 2 * x1 + 4 * x2 + kernel_size4 * x1 + x2 * kernel_size4 * kernel_size4 + 4 * kernel_size4 * x2) // (4 + kernel_size4 * kernel_size4 + 4 * kernel_size4)) % (2 + kernel_size5)))) +
            2 * x5 * kernel_size4 * kernel_size4 + 4 * kernel_size4 * (((((x0 + 2 * x1 + 4 * x2 + kernel_size4 * x1 + x2 * kernel_size4 * kernel_size4 + 4 * kernel_size4 * x2) // (4 + kernel_size4 * kernel_size4 + 4 * kernel_size4)) % (2 + kernel_size5)))) +
            4 * kernel_size5 * x5 + 8 * kernel_size4 * x5 + kernel_size5 * x5 * kernel_size4 * kernel_size4 + 4 * kernel_size4 * kernel_size5 * x5
        ), 
        mask, 
        eviction_policy='evict_last'
    )

    tmp3 = tl.load(input_ptr1 + (x8 // 16), mask, eviction_policy='evict_last')
    tmp5 = tl.load(input_ptr2 + (x8 // 16), mask, eviction_policy='evict_last')
    tmp13 = tl.load(input_ptr3 + (x3), mask, eviction_policy='evict_last')
    tmp15 = tl.load(input_ptr4 + (x3), mask, eviction_policy='evict_last')

    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = tmp2 - tmp3

    tmp6 = 128 + 32 * kernel_size4 * kernel_size4 + 64 * kernel_size5 + 128 * kernel_size4 + 16 * kernel_size5 * kernel_size4 * kernel_size4 + 64 * kernel_size4 * kernel_size5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7

    epsilon = 1e-05
    tmp10 = tmp8 + epsilon
    tmp11 = tl.extra.cuda.libdevice.rsqrt(tmp10)
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15

    tl.store(output_ptr0 + (x9), tmp16, mask)