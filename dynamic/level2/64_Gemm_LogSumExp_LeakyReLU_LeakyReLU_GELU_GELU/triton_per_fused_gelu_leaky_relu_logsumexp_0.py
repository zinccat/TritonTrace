# From: 64_Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_leaky_relu_logsumexp_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 2, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_gelu_leaky_relu_logsumexp_0(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 512*x0), None)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp1, 0))
    tmp4 = tl_math.abs(tmp3)
    tmp5 = float("inf")
    tmp6 = tmp4 == tmp5
    tmp7 = 0.0
    tmp8 = tl.where(tmp6, tmp7, tmp3)
    tmp9 = tmp0 - tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tl_math.log(tmp13)
    tmp15 = tmp14 + tmp8
    tmp16 = tmp15 > tmp7
    tmp17 = 0.01
    tmp18 = tmp15 * tmp17
    tmp19 = tl.where(tmp16, tmp15, tmp18)
    tmp20 = tmp19 > tmp7
    tmp21 = tmp19 * tmp17
    tmp22 = tl.where(tmp20, tmp19, tmp21)
    tmp23 = 0.5
    tmp24 = tmp22 * tmp23
    tmp25 = 0.7071067811865476
    tmp26 = tmp22 * tmp25
    tmp27 = libdevice.erf(tmp26)
    tmp28 = 1.0
    tmp29 = tmp27 + tmp28
    tmp30 = tmp24 * tmp29
    tmp31 = tmp30 * tmp23
    tmp32 = tmp30 * tmp25
    tmp33 = libdevice.erf(tmp32)
    tmp34 = tmp33 + tmp28
    tmp35 = tmp31 * tmp34
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp15, None)
    tl.store(out_ptr1 + (x0), tmp35, None)
