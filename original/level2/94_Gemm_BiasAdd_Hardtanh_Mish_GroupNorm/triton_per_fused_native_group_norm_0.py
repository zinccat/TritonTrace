# From: 94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (r2 + (32*x3)), None)
    tmp1 = tl.load(in_ptr1 + (r2 + (32*x0)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = -1.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 1.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = 20.0
    tmp8 = tmp6 > tmp7
    tmp9 = tl_math.exp(tmp6)
    tmp10 = libdevice.log1p(tmp9)
    tmp11 = tl.where(tmp8, tmp6, tmp10)
    tmp12 = libdevice.tanh(tmp11)
    tmp13 = tmp6 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp18 = tl.sum(tmp16, 1)[:, None]
    tmp19 = tl.full([XBLOCK, 1], 32, tl.int32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 / tmp20
    tmp22 = tmp14 - tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp26 = tl.sum(tmp24, 1)[:, None]
    tmp27 = 32.0
    tmp28 = tmp26 / tmp27
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tl.store(out_ptr2 + (x3), tmp31, None)
    tl.store(out_ptr0 + (x3), tmp21, None)
    tl.store(out_ptr1 + (x3), tmp26, None)
