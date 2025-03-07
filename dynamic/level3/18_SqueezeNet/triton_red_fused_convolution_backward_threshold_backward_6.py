# From: 18_SqueezeNet

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[256, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_threshold_backward_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + x0 + (x0*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks0,  2)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks0,  2)),  2)),  2)))) + (2*x0*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks0,  2)),  2)),  2)))), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (r1 + x0 + (x0*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks0,  2)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks0,  2)),  2)),  2)))) + (2*x0*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks0,  2)),  2)),  2)))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
        tl.store(out_ptr0 + (r1 + x0 + (x0*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks0,  2)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks0,  2)),  2)),  2)))) + (2*x0*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks0,  2)),  2)),  2)))), tmp3, rmask & xmask)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp5, xmask)
