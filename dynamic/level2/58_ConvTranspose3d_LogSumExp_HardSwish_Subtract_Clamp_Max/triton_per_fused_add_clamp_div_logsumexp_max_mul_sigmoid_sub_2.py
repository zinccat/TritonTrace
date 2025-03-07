# From: 58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp_Max

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16777216, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i64', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clamp_div_logsumexp_max_mul_sigmoid_sub_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_clamp_div_logsumexp_max_mul_sigmoid_sub_2(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp1 = tl_math.log(tmp0)
    tmp3 = tl_math.abs(tmp2)
    tmp4 = float("inf")
    tmp5 = tmp3 == tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp2)
    tmp8 = tmp1 + tmp7
    tmp9 = 3.0
    tmp10 = tmp8 + tmp9
    tmp11 = tl.sigmoid(tmp10)
    tmp12 = tmp8 * tmp11
    tmp13 = 0.16666666666666666
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 - tmp15
    tmp17 = -1.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 1.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(xmask, tmp21, float("-inf"))
    tmp24 = triton_helpers.max2(tmp23, 1)[:, None]
    tmp26 = tl.broadcast_to(rindex, tmp23.shape)
    tmp25_val, tmp25_idx = triton_helpers.max_with_index(tmp23, tmp26, 1)
    tmp25 = tmp25_idx[:, None]
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr0 + (x0), tmp24, xmask)
    tl.store(out_ptr1 + (x0), tmp25, xmask)
