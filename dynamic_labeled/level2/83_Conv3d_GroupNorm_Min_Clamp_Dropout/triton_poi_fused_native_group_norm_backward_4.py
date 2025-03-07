# From: 83_Conv3d_GroupNorm_Min_Clamp_Dropout

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_backward_4(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, 
    kernel_size0, kernel_size1, kernel_size2, kernel_size3, num_elements, 
    XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    x_mod_k0 = index % kernel_size0
    x_div_k0 = index // kernel_size0
    x_div_k3 = index // kernel_size3
    x_mod_16 = (index // kernel_size0) % 16
    original_index = index

    tmp0 = tl.load(
        in_ptr0 + (
            (-8) * x_div_k0 + 
            (-2) * ((x_mod_k0 // ((-2) + kernel_size2)) % ((-2) + kernel_size2)) + 
            4 * triton_helpers.div_floor_integer(x_mod_k0, 4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2)) + 
            kernel_size2 * ((x_mod_k0 // ((-2) + kernel_size2)) % ((-2) + kernel_size2)) + 
            kernel_size2 * kernel_size2 * triton_helpers.div_floor_integer(x_mod_k0, 4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2)) + 
            ((-4) * kernel_size2 * triton_helpers.div_floor_integer(x_mod_k0, 4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2))) + 
            ((-2) * x_div_k0 * kernel_size2 * kernel_size2) + 
            4 * kernel_size1 * x_div_k0 + 
            8 * kernel_size2 * x_div_k0 + 
            kernel_size1 * x_div_k0 * kernel_size2 * kernel_size2 + 
            ((-4) * kernel_size1 * kernel_size2 * x_div_k0) + 
            (x_mod_k0 % ((-2) + kernel_size2))
        ), 
        mask, 
        eviction_policy='evict_last'
    )

    tmp9 = tl.load(
        in_ptr1 + (
            (-8) * x_div_k0 + 
            (-2) * ((x_mod_k0 // ((-2) + kernel_size2)) % ((-2) + kernel_size2)) + 
            4 * triton_helpers.div_floor_integer(x_mod_k0, 4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2)) + 
            kernel_size2 * ((x_mod_k0 // ((-2) + kernel_size2)) % ((-2) + kernel_size2)) + 
            kernel_size2 * kernel_size2 * triton_helpers.div_floor_integer(x_mod_k0, 4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2)) + 
            ((-4) * kernel_size2 * triton_helpers.div_floor_integer(x_mod_k0, 4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2))) + 
            ((-2) * x_div_k0 * kernel_size2 * kernel_size2) + 
            4 * kernel_size1 * x_div_k0 + 
            8 * kernel_size2 * x_div_k0 + 
            kernel_size1 * x_div_k0 * kernel_size2 * kernel_size2 + 
            ((-4) * kernel_size1 * kernel_size2 * x_div_k0) + 
            (x_mod_k0 % ((-2) + kernel_size2))
        ), 
        mask, 
        eviction_policy='evict_last'
    )

    tmp10 = tl.load(
        in_ptr2 + (
            (-8) * x_div_k0 + 
            (-2) * ((x_mod_k0 // ((-2) + kernel_size2)) % ((-2) + kernel_size2)) + 
            4 * triton_helpers.div_floor_integer(x_mod_k0, 4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2)) + 
            kernel_size2 * ((x_mod_k0 // ((-2) + kernel_size2)) % ((-2) + kernel_size2)) + 
            kernel_size2 * kernel_size2 * triton_helpers.div_floor_integer(x_mod_k0, 4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2)) + 
            ((-4) * kernel_size2 * triton_helpers.div_floor_integer(x_mod_k0, 4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2))) + 
            ((-2) * x_div_k0 * kernel_size2 * kernel_size2) + 
            4 * kernel_size1 * x_div_k0 + 
            8 * kernel_size2 * x_div_k0 + 
            kernel_size1 * x_div_k0 * kernel_size2 * kernel_size2 + 
            ((-4) * kernel_size1 * kernel_size2 * x_div_k0) + 
            (x_mod_k0 % ((-2) + kernel_size2))
        ), 
        mask, 
        eviction_policy='evict_last'
    ).to(tl.int1)

    tmp20 = tl.load(in_ptr3 + x_div_k3, mask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + x_mod_16, mask, eviction_policy='evict_last')
    tmp24 = tl.load(in_out_ptr0 + original_index, mask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + x_div_k3, mask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + x_div_k3, mask, eviction_policy='evict_last')

    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 == tmp1
    tmp4 = triton_helpers.minimum(tmp0, tmp1)
    tmp5 = tmp4 >= tmp1
    tmp6 = 1.0
    tmp7 = tmp4 <= tmp6
    tmp8 = tmp5 & tmp7
    tmp11 = tmp10.to(tl.float32)
    tmp12 = 1.25
    tmp13 = tmp11 * tmp12
    tmp14 = tmp9 * tmp13
    tmp15 = tl.where(tmp8, tmp14, tmp1)
    tmp16 = 0.5
    tmp17 = tmp15 * tmp16
    tmp18 = tl.where(tmp3, tmp17, tmp15)
    tmp19 = tl.where(tmp2, tmp1, tmp18)
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 * tmp22
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 + tmp26
    tmp29 = tmp27 + tmp28

    tl.store(in_out_ptr0 + original_index, tmp29, mask)