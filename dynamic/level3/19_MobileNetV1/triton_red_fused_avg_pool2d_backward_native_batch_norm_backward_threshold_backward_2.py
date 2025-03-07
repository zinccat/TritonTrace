# From: 19_MobileNetV1

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32', 5: 'i32', 6: 'i32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex % ks0
        r6 = (rindex // ks0)
        r1 = rindex % ks2
        r2 = (rindex // ks2) % ks2
        r3 = (rindex // ks3)
        tmp0 = tl.load(in_ptr0 + (r5 + x0 + (1024*r6) + (x0*((triton_helpers.div_floor_integer((-1) + ks1,  32))*(triton_helpers.div_floor_integer((-1) + ks1,  32)))) + (2*x0*(triton_helpers.div_floor_integer((-1) + ks1,  32))) + (1024*r6*((triton_helpers.div_floor_integer((-1) + ks1,  32))*(triton_helpers.div_floor_integer((-1) + ks1,  32)))) + (2048*r6*(triton_helpers.div_floor_integer((-1) + ks1,  32)))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (1024*r3) + (x0*((triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))*(triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7)))) + ((triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))*((((0) * ((0) >= ((r2 // 7))) + ((r2 // 7)) * (((r2 // 7)) > (0)))) * ((((0) * ((0) >= ((r2 // 7))) + ((r2 // 7)) * (((r2 // 7)) > (0)))) <= ((-1) + ((1 + (r2 // 7)) * ((1 + (r2 // 7)) <= (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7)))) + (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))) * ((1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))) < (1 + (r2 // 7)))))) + ((-1) + ((1 + (r2 // 7)) * ((1 + (r2 // 7)) <= (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7)))) + (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))) * ((1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))) < (1 + (r2 // 7))))) * (((-1) + ((1 + (r2 // 7)) * ((1 + (r2 // 7)) <= (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7)))) + (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))) * ((1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))) < (1 + (r2 // 7))))) < (((0) * ((0) >= ((r2 // 7))) + ((r2 // 7)) * (((r2 // 7)) > (0))))))) + (2*x0*(triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))) + (1024*r3*((triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))*(triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7)))) + (2048*r3*(triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))) + ((((0) * ((0) >= ((r1 // 7))) + ((r1 // 7)) * (((r1 // 7)) > (0)))) * ((((0) * ((0) >= ((r1 // 7))) + ((r1 // 7)) * (((r1 // 7)) > (0)))) <= ((-1) + ((1 + (r1 // 7)) * ((1 + (r1 // 7)) <= (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7)))) + (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))) * ((1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))) < (1 + (r1 // 7)))))) + ((-1) + ((1 + (r1 // 7)) * ((1 + (r1 // 7)) <= (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7)))) + (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))) * ((1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))) < (1 + (r1 // 7))))) * (((-1) + ((1 + (r1 // 7)) * ((1 + (r1 // 7)) <= (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7)))) + (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))) * ((1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))) < (1 + (r1 // 7))))) < (((0) * ((0) >= ((r1 // 7))) + ((r1 // 7)) * (((r1 // 7)) > (0)))))) + ((((0) * ((0) >= ((r2 // 7))) + ((r2 // 7)) * (((r2 // 7)) > (0)))) * ((((0) * ((0) >= ((r2 // 7))) + ((r2 // 7)) * (((r2 // 7)) > (0)))) <= ((-1) + ((1 + (r2 // 7)) * ((1 + (r2 // 7)) <= (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7)))) + (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))) * ((1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))) < (1 + (r2 // 7)))))) + ((-1) + ((1 + (r2 // 7)) * ((1 + (r2 // 7)) <= (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7)))) + (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))) * ((1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))) < (1 + (r2 // 7))))) * (((-1) + ((1 + (r2 // 7)) * ((1 + (r2 // 7)) <= (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7)))) + (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))) * ((1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))) < (1 + (r2 // 7))))) < (((0) * ((0) >= ((r2 // 7))) + ((r2 // 7)) * (((r2 // 7)) > (0))))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tmp3 / 49
        tmp5 = ((0) * ((0) >= ((r2 // 7))) + ((r2 // 7)) * (((r2 // 7)) > (0)))
        tmp6 = ((1 + (r2 // 7)) * ((1 + (r2 // 7)) <= (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7)))) + (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))) * ((1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))) < (1 + (r2 // 7))))
        tmp7 = tmp5 < tmp6
        tmp8 = ((0) * ((0) >= ((r1 // 7))) + ((r1 // 7)) * (((r1 // 7)) > (0)))
        tmp9 = ((1 + (r1 // 7)) * ((1 + (r1 // 7)) <= (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7)))) + (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))) * ((1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks1,  32)),  7))) < (1 + (r1 // 7))))
        tmp10 = tmp8 < tmp9
        tmp11 = tmp7 & tmp10
        tmp12 = tl.where(tmp11, tmp4, tmp1)
        tmp13 = tl.where(tmp2, tmp1, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp15, xmask)
