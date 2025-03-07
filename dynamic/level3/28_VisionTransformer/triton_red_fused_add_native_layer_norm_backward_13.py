# From: 28_VisionTransformer

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32', 7: 'i32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_13', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ks0, ks1, ks2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (x1*((3 + ks0 + (ks0*((ks1 // 16)*(ks1 // 16)))) // 4))
        tmp1 = ks0 + (ks0*((ks1 // 16)*(ks1 // 16)))
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*((r2 + (x1*((3 + ks0 + (ks0*((ks1 // 16)*(ks1 // 16)))) // 4))) % ks2)) + (512*(((r2 + (x1*((3 + ks0 + (ks0*((ks1 // 16)*(ks1 // 16)))) // 4))) // ks2) % ks0)) + (512*((ks1 // 16)*(ks1 // 16))*(((r2 + (x1*((3 + ks0 + (ks0*((ks1 // 16)*(ks1 // 16)))) // 4))) // ks2) % ks0))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (512*((r2 + (x1*((3 + ks0 + (ks0*((ks1 // 16)*(ks1 // 16)))) // 4))) % ks2)) + (512*(((r2 + (x1*((3 + ks0 + (ks0*((ks1 // 16)*(ks1 // 16)))) // 4))) // ks2) % ks0)) + (512*((ks1 // 16)*(ks1 // 16))*(((r2 + (x1*((3 + ks0 + (ks0*((ks1 // 16)*(ks1 // 16)))) // 4))) // ks2) % ks0))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.load(in_ptr2 + (x0 + (512*((r2 + (x1*((3 + ks0 + (ks0*((ks1 // 16)*(ks1 // 16)))) // 4))) % ks2)) + (512*(((r2 + (x1*((3 + ks0 + (ks0*((ks1 // 16)*(ks1 // 16)))) // 4))) // ks2) % ks0)) + (512*((ks1 // 16)*(ks1 // 16))*(((r2 + (x1*((3 + ks0 + (ks0*((ks1 // 16)*(ks1 // 16)))) // 4))) // ks2) % ks0))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp5 * tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask, tmp12, _tmp11)
        tmp13 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp14 = tl.where(tmp2, tmp5, tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, None)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp16, None)
