# From: 15_DenseNet121

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32', 6: 'i32', 7: 'i32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_12', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, ks0, ks1, ks2, ks3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128)
    x0 = xindex % 128
    tmp13_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (x1*(triton_helpers.div_floor_integer(3 + ks0 + (ks0*((triton_helpers.div_floor_integer((-1) + ks1,  4))*(triton_helpers.div_floor_integer((-1) + ks1,  4)))) + (2*ks0*(triton_helpers.div_floor_integer((-1) + ks1,  4))),  4)))
        tmp1 = ks0 + (ks0*((triton_helpers.div_floor_integer((-1) + ks1,  4))*(triton_helpers.div_floor_integer((-1) + ks1,  4)))) + (2*ks0*(triton_helpers.div_floor_integer((-1) + ks1,  4)))
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (128*(((r2 + (x1*(triton_helpers.div_floor_integer(3 + ks0 + (ks0*((triton_helpers.div_floor_integer((-1) + ks1,  4))*(triton_helpers.div_floor_integer((-1) + ks1,  4)))) + (2*ks0*(triton_helpers.div_floor_integer((-1) + ks1,  4))),  4)))) // ks3) % ks0)) + (x0*((triton_helpers.div_floor_integer((-1) + ks1,  4))*(triton_helpers.div_floor_integer((-1) + ks1,  4)))) + ((triton_helpers.div_floor_integer((-1) + ks1,  4))*(((r2 + (x1*(triton_helpers.div_floor_integer(3 + ks0 + (ks0*((triton_helpers.div_floor_integer((-1) + ks1,  4))*(triton_helpers.div_floor_integer((-1) + ks1,  4)))) + (2*ks0*(triton_helpers.div_floor_integer((-1) + ks1,  4))),  4)))) // ks2) % ks2)) + (2*x0*(triton_helpers.div_floor_integer((-1) + ks1,  4))) + (128*((triton_helpers.div_floor_integer((-1) + ks1,  4))*(triton_helpers.div_floor_integer((-1) + ks1,  4)))*(((r2 + (x1*(triton_helpers.div_floor_integer(3 + ks0 + (ks0*((triton_helpers.div_floor_integer((-1) + ks1,  4))*(triton_helpers.div_floor_integer((-1) + ks1,  4)))) + (2*ks0*(triton_helpers.div_floor_integer((-1) + ks1,  4))),  4)))) // ks3) % ks0)) + (256*(triton_helpers.div_floor_integer((-1) + ks1,  4))*(((r2 + (x1*(triton_helpers.div_floor_integer(3 + ks0 + (ks0*((triton_helpers.div_floor_integer((-1) + ks1,  4))*(triton_helpers.div_floor_integer((-1) + ks1,  4)))) + (2*ks0*(triton_helpers.div_floor_integer((-1) + ks1,  4))),  4)))) // ks3) % ks0)) + ((r2 + (x1*(triton_helpers.div_floor_integer(3 + ks0 + (ks0*((triton_helpers.div_floor_integer((-1) + ks1,  4))*(triton_helpers.div_floor_integer((-1) + ks1,  4)))) + (2*ks0*(triton_helpers.div_floor_integer((-1) + ks1,  4))),  4)))) % ks2) + (((r2 + (x1*(triton_helpers.div_floor_integer(3 + ks0 + (ks0*((triton_helpers.div_floor_integer((-1) + ks1,  4))*(triton_helpers.div_floor_integer((-1) + ks1,  4)))) + (2*ks0*(triton_helpers.div_floor_integer((-1) + ks1,  4))),  4)))) // ks2) % ks2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tl.full(tmp4.shape, 0, tmp4.dtype)
        tmp6 = tl.where(tmp2, tmp4, tmp5)
        tmp7 = 1.0
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp11 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_combine(
            tmp13_mean, tmp13_m2, tmp13_weight,
            tmp10, tmp11, tmp12
        )
        tmp13_mean = tl.where(rmask & xmask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(rmask & xmask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(rmask & xmask, tmp13_weight_next, tmp13_weight)
    tmp13_tmp, tmp14_tmp, tmp15_tmp = triton_helpers.welford(
        tmp13_mean, tmp13_m2, tmp13_weight, 1
    )
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tmp15 = tmp15_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tl.store(out_ptr2 + (x3), tmp15, xmask)
