# From: 60_ConvTranspose3d_Swish_GroupNorm_HardSwish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_mul_native_group_norm_backward_sigmoid_sigmoid_backward_4(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, 
    kernel_size_0, kernel_size_1, kernel_size_2, kernel_size_3, kernel_size_4, 
    xnumel, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < xnumel

    x_mod_k0 = x_index % kernel_size_0
    x_div_k0 = x_index // kernel_size_0
    x_div_k4 = x_index // kernel_size_4
    x_mod_k1 = (x_index // kernel_size_0) % 16
    x_full_index = x_index

    tmp0 = tl.load(
        in_ptr0 + (
            (-1) * x_div_k0 + 
            (-1) * ((x_mod_k0 // kernel_size_1) % kernel_size_1) + 
            (-4) * kernel_size_3 * (triton_helpers.div_floor_integer(x_mod_k0, 1 + (-4) * kernel_size_3 + 4 * kernel_size_3 * kernel_size_3)) + 
            (-4) * x_div_k0 * kernel_size_3 * kernel_size_3 + 
            2 * kernel_size_2 * x_div_k0 + 
            2 * kernel_size_3 * ((x_mod_k0 // kernel_size_1) % kernel_size_1) + 
            4 * kernel_size_3 * x_div_k0 + 
            4 * kernel_size_3 * kernel_size_3 * (triton_helpers.div_floor_integer(x_mod_k0, 1 + (-4) * kernel_size_3 + 4 * kernel_size_3 * kernel_size_3)) + 
            (-8) * kernel_size_2 * kernel_size_3 * x_div_k0 + 
            8 * kernel_size_2 * x_div_k0 * kernel_size_3 * kernel_size_3 + 
            (triton_helpers.div_floor_integer(x_mod_k0, 1 + (-4) * kernel_size_3 + 4 * kernel_size_3 * kernel_size_3)) + 
            (x_mod_k0 % kernel_size_1)
        ), 
        x_mask, 
        eviction_policy='evict_last'
    )

    tmp5 = tl.load(
        in_ptr1 + (
            (-1) * x_div_k0 + 
            (-1) * ((x_mod_k0 // kernel_size_1) % kernel_size_1) + 
            (-4) * kernel_size_3 * (triton_helpers.div_floor_integer(x_mod_k0, 1 + (-4) * kernel_size_3 + 4 * kernel_size_3 * kernel_size_3)) + 
            (-4) * x_div_k0 * kernel_size_3 * kernel_size_3 + 
            2 * kernel_size_2 * x_div_k0 + 
            2 * kernel_size_3 * ((x_mod_k0 // kernel_size_1) % kernel_size_1) + 
            4 * kernel_size_3 * x_div_k0 + 
            4 * kernel_size_3 * kernel_size_3 * (triton_helpers.div_floor_integer(x_mod_k0, 1 + (-4) * kernel_size_3 + 4 * kernel_size_3 * kernel_size_3)) + 
            (-8) * kernel_size_2 * kernel_size_3 * x_div_k0 + 
            8 * kernel_size_2 * x_div_k0 * kernel_size_3 * kernel_size_3 + 
            (triton_helpers.div_floor_integer(x_mod_k0, 1 + (-4) * kernel_size_3 + 4 * kernel_size_3 * kernel_size_3)) + 
            (x_mod_k0 % kernel_size_1)
        ), 
        x_mask, 
        eviction_policy='evict_last'
    )

    tmp14 = tl.load(in_ptr2 + (x_div_k4), x_mask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x_mod_k1), x_mask, eviction_policy='evict_last')
    tmp18 = tl.load(
        in_ptr4 + (
            (-1) * x_div_k0 + 
            (-1) * ((x_mod_k0 // kernel_size_1) % kernel_size_1) + 
            (-4) * kernel_size_3 * (triton_helpers.div_floor_integer(x_mod_k0, 1 + (-4) * kernel_size_3 + 4 * kernel_size_3 * kernel_size_3)) + 
            (-4) * x_div_k0 * kernel_size_3 * kernel_size_3 + 
            2 * kernel_size_2 * x_div_k0 + 
            2 * kernel_size_3 * ((x_mod_k0 // kernel_size_1) % kernel_size_1) + 
            4 * kernel_size_3 * x_div_k0 + 
            4 * kernel_size_3 * kernel_size_3 * (triton_helpers.div_floor_integer(x_mod_k0, 1 + (-4) * kernel_size_3 + 4 * kernel_size_3 * kernel_size_3)) + 
            (-8) * kernel_size_2 * kernel_size_3 * x_div_k0 + 
            8 * kernel_size_2 * x_div_k0 * kernel_size_3 * kernel_size_3 + 
            (triton_helpers.div_floor_integer(x_mod_k0, 1 + (-4) * kernel_size_3 + 4 * kernel_size_3 * kernel_size_3)) + 
            (x_mod_k0 % kernel_size_1)
        ), 
        x_mask, 
        eviction_policy='evict_last'
    )

    tmp21 = tl.load(in_ptr5 + (x_div_k4), x_mask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr6 + (x_div_k0 // 4), x_mask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr4 + (x_full_index), x_mask, eviction_policy='evict_last')

    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = 0.3333333333333333
    tmp7 = tmp0 * tmp6
    tmp8 = 0.5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp5 * tmp9
    tmp11 = tl.where(tmp4, tmp10, tmp5)
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp12, tmp11)

    tmp16 = tmp14 * tmp15
    tmp17 = tmp13 * tmp16

    tmp19 = tl.sigmoid(tmp18)
    tmp20 = tmp19 * tmp18

    tmp22 = tmp21 * tmp14
    tmp23 = tmp22 * tmp14
    tmp24 = tmp23 * tmp14

    tmp25 = 2.0
    tmp26 = kernel_size_3
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 * tmp27
    tmp29 = -1.0
    tmp30 = tmp29 + tmp28
    tmp31 = tl.extra.cuda.libdevice.pow(tmp30, tmp25)
    tmp32 = 4.0
    tmp33 = tmp32 * tmp31

    tmp34 = kernel_size_2
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp25 * tmp35
    tmp37 = tmp29 + tmp36
    tmp38 = tmp33 * tmp37
    tmp39 = tmp38.to(tl.float64)

    tmp40 = tl.full([1], 1.0, tl.float64)
    tmp41 = tmp40 / tmp39
    tmp42 = tmp41.to(tl.float32)

    tmp43 = tmp24 * tmp42
    tmp44 = tmp20 * tmp43
    tmp45 = tmp17 + tmp44
    tmp47 = tmp45 + tmp46

    tmp49 = tl.sigmoid(tmp48)
    tmp50 = tmp47 * tmp49
    tmp51 = tmp47 * tmp48
    tmp52 = 1.0
    tmp53 = tmp52 - tmp49
    tmp54 = tmp49 * tmp53
    tmp55 = tmp51 * tmp54
    tmp56 = tmp50 + tmp55

    tl.store(in_out_ptr0 + (x_full_index), tmp56, x_mask)