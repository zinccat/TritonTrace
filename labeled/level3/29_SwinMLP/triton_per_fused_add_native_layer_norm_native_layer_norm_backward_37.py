# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_37(
    in_out_ptr, input_ptr0, input_ptr1, input_ptr2, input_ptr3, output_ptr2, output_ptr3, xnumel, rnumel
):
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[:]
    r_mask = r_index < rnumel
    r2 = r_index
    x3 = x_index
    x0 = (x_index % 196)
    x1 = x_index // 196

    tmp0 = tl.load(in_out_ptr + (r2 + 384 * x3), r_mask, other=0.0)
    tmp1 = tl.load(
        input_ptr0 + (
            32 * (((x0 % 14) % 7)) + 224 * (((x0 // 14) % 7)) + 1568 * (r2 // 32) +
            18816 * (((x0 % 14)) // 7) + 37632 * (x0 // 98) + 75264 * x1 + (r2 % 32)
        ),
        r_mask,
        other=0.0
    )
    tmp2 = tl.load(
        input_ptr1 + (7 * ((x0 // 14) % 7) + 49 * (r2 // 32) + ((x0 % 14) % 7)),
        r_mask,
        eviction_policy='evict_last',
        other=0.0
    )
    tmp5 = tl.load(in_out_ptr + (r2 + 384 * x3), r_mask, other=0.0)
    tmp6 = tl.load(input_ptr3 + (r2), r_mask, eviction_policy='evict_last', other=0.0)

    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7

    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tl.where(r_mask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(r_mask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 384, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(r_mask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 384.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.extra.cuda.libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp32 = 0.0026041666666666665
    tmp33 = tmp30 * tmp32

    tl.store(in_out_ptr + (r2 + 384 * x3), tmp8, r_mask)
    tl.store(output_ptr2 + (r2 + 384 * x3), tmp31, r_mask)
    tl.store(output_ptr3 + (x3), tmp33, None)