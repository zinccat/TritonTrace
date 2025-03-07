# From: 22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clamp_logsumexp_mish_mul_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 2, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_clamp_logsumexp_mish_mul_0(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 1024*x0), None)
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2 + tmp2
    tmp4 = -10.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = 10.0
    tmp7 = triton_helpers.minimum(tmp5, tmp6)
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp8, 0))
    tmp11 = tl_math.abs(tmp10)
    tmp12 = float("inf")
    tmp13 = tmp11 == tmp12
    tmp14 = 0.0
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = tmp7 - tmp15
    tmp17 = tl_math.exp(tmp16)
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp21 = tl_math.log(tmp20)
    tmp22 = tmp21 + tmp15
    tmp23 = 20.0
    tmp24 = tmp22 > tmp23
    tmp25 = tl_math.exp(tmp22)
    tmp26 = libdevice.log1p(tmp25)
    tmp27 = tl.where(tmp24, tmp22, tmp26)
    tmp28 = libdevice.tanh(tmp27)
    tmp29 = tmp22 * tmp28
    tmp30 = tmp22 * tmp29
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp22, None)
    tl.store(out_ptr1 + (x0), tmp30, None)
