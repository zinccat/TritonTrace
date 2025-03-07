# From: 31_VisionAttention

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_native_layer_norm_8poi_fused_add_native_layer_norm_8(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr
):
    xnumel = 256
    y_offset = tl.program_id(1) * YBLOCK
    y_index = y_offset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    x3 = x_index
    y0 = y_index
    x1 = (x_index % 128)
    x2 = x_index // 128

    tmp0 = tl.load(in_out_ptr0 + (x3 + 256 * y0), x_mask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x1), x_mask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + 16384 * x3), x_mask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + 2 * y0), x_mask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + 2 * y0), x_mask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), x_mask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), x_mask, eviction_policy='evict_last')

    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5

    tmp8 = 128.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = tl.extra.cuda.libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16

    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3 + 256 * y0), tmp13, x_mask)
    tl.store(out_ptr0 + (x3 + 256 * y0), tmp17, x_mask)