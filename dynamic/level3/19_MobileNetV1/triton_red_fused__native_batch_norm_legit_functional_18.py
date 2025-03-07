# From: 19_MobileNetV1

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32', 10: 'i32', 11: 'i32', 12: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_18', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, ks0, ks1, ks2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex % ks0
        r4 = (rindex // ks0)
        tmp0 = tl.load(in_ptr0 + (r3 + x0 + (512*r4) + (x0*((triton_helpers.div_floor_integer((-1) + ks1,  16))*(triton_helpers.div_floor_integer((-1) + ks1,  16)))) + (2*x0*(triton_helpers.div_floor_integer((-1) + ks1,  16))) + (512*r4*((triton_helpers.div_floor_integer((-1) + ks1,  16))*(triton_helpers.div_floor_integer((-1) + ks1,  16)))) + (1024*r4*(triton_helpers.div_floor_integer((-1) + ks1,  16)))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(out_ptr1 + (x0), tmp3, xmask)
    tmp13 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = ks2 + (ks2*((triton_helpers.div_floor_integer((-1) + ks1,  16))*(triton_helpers.div_floor_integer((-1) + ks1,  16)))) + (2*ks2*(triton_helpers.div_floor_integer((-1) + ks1,  16)))
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp3 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = 0.1
    tmp12 = tmp2 * tmp11
    tmp14 = 0.9
    tmp15 = tmp13 * tmp14
    tmp16 = tmp12 + tmp15
    tmp17 = (((512*ks2) + (512*ks2*((triton_helpers.div_floor_integer((-1) + ks1,  16))*(triton_helpers.div_floor_integer((-1) + ks1,  16)))) + (1024*ks2*(triton_helpers.div_floor_integer((-1) + ks1,  16)))) / 512) / ((-1.00000000000000) + (((512*ks2) + (512*ks2*((triton_helpers.div_floor_integer((-1) + ks1,  16))*(triton_helpers.div_floor_integer((-1) + ks1,  16)))) + (1024*ks2*(triton_helpers.div_floor_integer((-1) + ks1,  16)))) / 512))
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp7 * tmp18
    tmp20 = tmp19 * tmp11
    tmp22 = tmp21 * tmp14
    tmp23 = tmp20 + tmp22
    tl.store(out_ptr2 + (x0), tmp10, xmask)
    tl.store(out_ptr4 + (x0), tmp16, xmask)
    tl.store(out_ptr6 + (x0), tmp23, xmask)
