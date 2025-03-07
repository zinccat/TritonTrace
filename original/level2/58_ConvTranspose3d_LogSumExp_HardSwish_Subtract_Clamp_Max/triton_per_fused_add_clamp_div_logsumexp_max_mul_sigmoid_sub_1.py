# From: 58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp_Max

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[16777216, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i64', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clamp_div_logsumexp_max_mul_sigmoid_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 15748992
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x2 = (xindex // 123039)
    x5 = xindex % 123039
    x0 = xindex % 3969
    x4 = (xindex // 3969)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x5 + (123039*r3) + (1968624*x2)), xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r3), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tl_math.abs(tmp4)
    tmp6 = float("inf")
    tmp7 = tmp5 == tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp7, tmp8, tmp4)
    tmp10 = tmp0 - tmp9
    tmp11 = tl_math.exp(tmp10)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl_math.log(tmp15)
    tmp17 = tmp16 + tmp9
    tmp18 = 3.0
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sigmoid(tmp19)
    tmp21 = tmp17 * tmp20
    tmp22 = 0.16666666666666666
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 - tmp24
    tmp26 = -1.0
    tmp27 = triton_helpers.maximum(tmp25, tmp26)
    tmp28 = 1.0
    tmp29 = triton_helpers.minimum(tmp27, tmp28)
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
    tmp32 = tl.where(xmask, tmp30, float("-inf"))
    tmp33 = triton_helpers.max2(tmp32, 1)[:, None]
    tmp35 = tl.broadcast_to(rindex, tmp32.shape)
    _, tmp34_tmp = triton_helpers.max_with_index(tmp32, tmp35, 1)
    tmp34 = tmp34_tmp[:, None]
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0 + (4000*x4)), tmp17, xmask)
    tl.store(out_ptr1 + (x6), tmp33, xmask)
    tl.store(out_ptr2 + (x0 + (3984*x4)), tmp34, xmask)
