# From: 29_SwinMLP

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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_53', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp12 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr2 + (r2 + (192*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = 4 + (x0 // 28)
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 35, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = 4 + (x0 % 28)
        tmp6 = tmp5 >= tmp1
        tmp7 = tmp5 < tmp3
        tmp8 = tmp2 & tmp4
        tmp9 = tmp8 & tmp6
        tmp10 = tmp9 & tmp7
        tmp11 = tl.load(in_ptr0 + ((32*((4 + (x0 % 28)) % 7)) + (224*((4 + (x0 // 28)) % 7)) + (1568*(r2 // 32)) + (9408*((4 + (x0 % 28)) // 7)) + (47040*(triton_helpers.div_floor_integer(4 + (x0 // 28),  7))) + (235200*x1) + (r2 % 32)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
        tmp18 = tmp13 * tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tmp23 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp22 = tl.load(in_out_ptr0 + (r2 + (192*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp36 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp41 = tl.load(in_ptr2 + (r2 + (192*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp24 = 4 + (x0 // 28)
        tmp25 = tl.full([1, 1], 0, tl.int64)
        tmp26 = tmp24 >= tmp25
        tmp27 = tl.full([1, 1], 35, tl.int64)
        tmp28 = tmp24 < tmp27
        tmp29 = 4 + (x0 % 28)
        tmp30 = tmp29 >= tmp25
        tmp31 = tmp29 < tmp27
        tmp32 = tmp26 & tmp28
        tmp33 = tmp32 & tmp30
        tmp34 = tmp33 & tmp31
        tmp35 = tl.load(in_ptr0 + ((32*((4 + (x0 % 28)) % 7)) + (224*((4 + (x0 // 28)) % 7)) + (1568*(r2 // 32)) + (9408*((4 + (x0 % 28)) // 7)) + (47040*(triton_helpers.div_floor_integer(4 + (x0 // 28),  7))) + (235200*x1) + (r2 % 32)), rmask & tmp34 & xmask, eviction_policy='evict_first', other=0.0)
        tmp37 = tmp35 * tmp36
        tmp38 = 192.0
        tmp39 = tmp37 * tmp38
        tmp40 = tmp39 - tmp15
        tmp42 = tmp41 * tmp20
        tmp43 = tmp40 - tmp42
        tmp44 = tmp23 * tmp43
        tmp45 = tmp22 + tmp44
        tl.store(in_out_ptr0 + (r2 + (192*x3)), tmp45, rmask & xmask)
