# From: 29_SwinMLP

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7840
    rnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = (xindex % 784)
    x1 = xindex // 784
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + 192*x3), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (32*(((4 + ((x0 % 28))) % 7)) + 224*(((4 + (x0 // 28)) % 7)) + 1568*(r2 // 32) + 9408*((4 + ((x0 % 28))) // 7) + 47040*(triton_helpers.div_floor_integer(4 + (x0 // 28),  7)) + 235200*x1 + ((r2 % 32))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (7*(((4 + (x0 // 28)) % 7)) + 49*(r2 // 32) + (((4 + ((x0 % 28))) % 7))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp9 = tl.load(in_ptr0 + (r2 + 192*x3), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr1 + (32*(((4 + ((x0 % 28))) % 7)) + 224*(((4 + (x0 // 28)) % 7)) + 1568*(r2 // 32) + 9408*((4 + ((x0 % 28))) // 7) + 47040*(triton_helpers.div_floor_integer(4 + (x0 // 28),  7)) + 235200*x1 + ((r2 % 32))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr2 + (7*(((4 + (x0 // 28)) % 7)) + 49*(r2 // 32) + (((4 + ((x0 % 28))) % 7))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 + tmp11
        tmp13 = tmp9 + tmp12
        tmp14 = tmp13 - tmp6
        tmp15 = 192.0
        tmp16 = tmp7 / tmp15
        tmp17 = 1e-05
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tl.store(out_ptr2 + (r2 + 192*x3), tmp20, rmask & xmask)
        tl.store(out_ptr3 + (r2 + 192*x3), tmp24, rmask & xmask)
    tmp25 = 192.0
    tmp26 = tmp7 / tmp25
    tmp27 = 1e-05
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = 0.005208333333333333
    tmp31 = tmp29 * tmp30
    tl.store(out_ptr4 + (x3), tmp31, xmask)
