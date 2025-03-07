# From: 27_Conv3d_HardSwish_ReLU_Softmax_Mean

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2097152, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__softmax_backward_data_div_hardswish_relu_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax__softmax_backward_data_div_hardswish_relu_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, ks1, ks2, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x4 = xindex // ks0
    x3 = (xindex % ks0)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + 16*x4), xmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr1 + (x3 + ((-128)*x4) + ((-8)*r2) + ((-32)*x4*ks2*ks2) + ((-2)*r2*ks2*ks2) + 4*ks1*r2 + 8*ks2*r2 + 64*ks1*x4 + 128*ks2*x4 + ks1*r2*ks2*ks2 + ((-64)*ks1*ks2*x4) + ((-4)*ks1*ks2*r2) + 16*ks1*x4*ks2*ks2), xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr2 + (x5), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr3 + (x5), xmask, eviction_policy='evict_last')
    tmp1 = ks0
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 / tmp2
    tmp5 = 3.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = 6.0
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tmp11 = tmp4 * tmp10
    tmp12 = 0.16666666666666666
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full([1, 1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tmp17 = tmp15 - tmp16
    tmp18 = tl_math.exp(tmp17)
    tmp20 = tmp18 / tmp19
    tmp21 = tmp3 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp25, xmask)
