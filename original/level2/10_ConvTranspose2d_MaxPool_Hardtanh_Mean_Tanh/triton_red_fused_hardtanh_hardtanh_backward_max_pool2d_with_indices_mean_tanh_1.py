# From: 10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i8', 3: '*i1', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_hardtanh_backward_max_pool2d_with_indices_mean_tanh_1', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 16
        r2 = (rindex // 16)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + ((2*r1) + (64*r2) + (1024*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr0 + (1 + (2*r1) + (64*r2) + (1024*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr0 + (32 + (2*r1) + (64*r2) + (1024*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr0 + (33 + (2*r1) + (64*r2) + (1024*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp1 > tmp0
        tmp3 = tl.full([1, 1], 1, tl.int8)
        tmp4 = tl.full([1, 1], 0, tl.int8)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = triton_helpers.maximum(tmp1, tmp0)
        tmp8 = tmp7 > tmp6
        tmp9 = tl.full([1, 1], 2, tl.int8)
        tmp10 = tl.where(tmp8, tmp9, tmp5)
        tmp11 = triton_helpers.maximum(tmp7, tmp6)
        tmp13 = tmp12 > tmp11
        tmp14 = tl.full([1, 1], 3, tl.int8)
        tmp15 = tl.where(tmp13, tmp14, tmp10)
        tmp16 = triton_helpers.maximum(tmp12, tmp11)
        tmp17 = -1.0
        tmp18 = tmp16 <= tmp17
        tmp19 = 1.0
        tmp20 = tmp16 >= tmp19
        tmp21 = tmp18 | tmp20
        tmp22 = triton_helpers.maximum(tmp16, tmp17)
        tmp23 = triton_helpers.minimum(tmp22, tmp19)
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask, tmp26, _tmp25)
        tl.store(out_ptr0 + (r3 + (256*x0)), tmp15, rmask)
        tl.store(out_ptr1 + (r3 + (256*x0)), tmp21, rmask)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tmp27 = 256.0
    tmp28 = tmp25 / tmp27
    tmp29 = libdevice.tanh(tmp28)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp29, None)
