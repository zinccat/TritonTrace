# From: 29_SwinMLP

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_33', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (x1*((15 + (196*ks0)) // 16))
        tmp1 = 196*ks0
        tmp2 = tmp0 < tmp1
        tmp3 = 4 + ((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) // 14) % 14)
        tmp4 = tl.full([1, 1], 0, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tl.full([1, 1], 21, tl.int64)
        tmp7 = tmp3 < tmp6
        tmp8 = 4 + (((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) % 14)
        tmp9 = tmp8 >= tmp4
        tmp10 = tmp8 < tmp6
        tmp11 = tmp5 & tmp7
        tmp12 = tmp11 & tmp9
        tmp13 = tmp12 & tmp10
        tmp14 = tmp13 & tmp2
        tmp15 = tl.load(in_ptr0 + ((32*((4 + (((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) % 14)) % 7)) + (224*((4 + ((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) // 14) % 14)) % 7)) + (1568*(x0 // 32)) + (18816*(((4 + (((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) % 14)) // 7) % 3)) + (56448*(((4 + ((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) // 14) % 14)) // 7) % 3)) + (169344*(((r2 + (x1*((15 + (196*ks0)) // 16))) // 196) % ks0)) + (x0 % 32)), rmask & tmp14, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr1 + (x0 + (384*((r2 + (x1*((15 + (196*ks0)) // 16))) % (196*ks0)))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp17 = tmp15 * tmp16
        tmp18 = tl.full(tmp17.shape, 0, tmp17.dtype)
        tmp19 = tl.where(tmp2, tmp17, tmp18)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask, tmp22, _tmp21)
        tmp23 = tl.full(tmp15.shape, 0, tmp15.dtype)
        tmp24 = tl.where(tmp2, tmp15, tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask, tmp27, _tmp26)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, None)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp26, None)
