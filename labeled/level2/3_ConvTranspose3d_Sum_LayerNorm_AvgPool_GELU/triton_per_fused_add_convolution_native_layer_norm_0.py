# From: 3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU

import triton
import triton.language as tl


@triton.jit
def triton_per_fused_add_convolution_native_layer_norm_0(
    in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, 
    out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = r_index
    x4 = x_index
    x1 = (x_index // 2048) % 64

    # Load and compute intermediate values
    tmp0 = tl.load(in_out_ptr0 + (r3 + (64 * x4)), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp26 = tl.load(in_ptr2 + (r3), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (r3), None, eviction_policy='evict_last')

    # Perform computations
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 + tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp10 = tl.sum(tmp8, 1)[:, None]
    tmp11 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 / tmp12
    tmp14 = tmp6 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.sum(tmp16, 1)[:, None]
    tmp19 = 64.0
    tmp20 = tmp18 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = tl.extra.cuda.libdevice.rsqrt(tmp22)
    tmp24 = tmp5 - tmp13
    tmp25 = tmp24 * tmp23
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28

    # Store results
    tl.store(in_out_ptr0 + (r3 + (64 * x4)), tmp2, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp23, None)
    tl.store(out_ptr1 + (r3 + (64 * x4)), tmp29, None)
    tl.store(out_ptr0 + (x4), tmp13, None)