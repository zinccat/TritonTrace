# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_19(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 7840
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_index = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_index < rnumel
    reduced_r_index = r_index
    expanded_x_index = x_index
    x_flat_index = x_index % 784
    x_channel_index = x_index // 784

    tmp0 = tl.load(in_ptr0 + (reduced_r_index + 192 * expanded_x_index), r_mask & x_mask, other=0.0)
    tmp1 = tl.load(
        in_ptr1 + (
            32 * (((x_flat_index % 28) % 7)) 
            + 224 * (((x_flat_index // 28) % 7)) 
            + 1568 * (reduced_r_index // 32) 
            + 9408 * (((x_flat_index % 28)) // 7) 
            + 37632 * (x_flat_index // 196) 
            + 150528 * x_channel_index 
            + ((reduced_r_index % 32))
        ),
        r_mask & x_mask,
        other=0.0
    )
    tmp2 = tl.load(
        in_ptr2 + (7 * (((x_flat_index // 28) % 7)) + 49 * (reduced_r_index // 32) + (((x_flat_index % 28)) % 7)),
        r_mask & x_mask,
        eviction_policy='evict_last',
        other=0.0
    )
    tmp5 = tl.load(in_out_ptr0 + (reduced_r_index + 192 * expanded_x_index), r_mask & x_mask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (reduced_r_index), r_mask, eviction_policy='evict_last', other=0.0)

    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7

    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tl.where(r_mask & x_mask, tmp9, 0)

    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(r_mask & x_mask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]

    tmp16 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17

    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])

    tmp23 = tl.where(r_mask & x_mask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]

    tmp25 = tmp8 - tmp18
    tmp26 = 192.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.extra.cuda.libdevice.rsqrt(tmp29)

    tmp31 = tmp25 * tmp30
    tmp32 = 0.005208333333333333
    tmp33 = tmp30 * tmp32

    tl.store(in_out_ptr0 + (reduced_r_index + 192 * expanded_x_index), tmp8, r_mask & x_mask)
    tl.store(out_ptr2 + (reduced_r_index + 192 * expanded_x_index), tmp31, r_mask & x_mask)
    tl.store(out_ptr3 + (expanded_x_index), tmp33, x_mask)