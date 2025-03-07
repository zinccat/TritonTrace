# From: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_backward_6poi_fused_native_group_norm_backward_6(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, 
    kernel_size0, kernel_size1, kernel_size2, kernel_size3, 
    xnumel, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < xnumel
    x_mod_k0 = x_index % kernel_size0
    x_div_k0 = x_index // kernel_size0
    x_mod_16 = x_div_k0 % 16
    x_div_k3 = x_index // kernel_size3
    x_full_index = x_index

    tmp0 = tl.load(
        in_ptr0 + (((-2) * (x_mod_k0 // kernel_size1)) + 4 * x_div_k0 + 
        kernel_size2 * (x_mod_k0 // kernel_size1) + x_div_k0 * kernel_size2 * kernel_size2 + 
        ((-4) * kernel_size2 * x_div_k0)), x_mask, eviction_policy='evict_last'
    )
    tmp1 = tl.load(in_ptr1 + (x_mod_16), x_mask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x_div_k3), x_mask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x_mod_16), x_mask, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (x_full_index), x_mask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x_div_k3), x_mask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x_div_k3), x_mask, eviction_policy='evict_last')

    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 * tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp12 = tmp10 + tmp11

    tl.store(in_out_ptr0 + (x_full_index), tmp12, x_mask)