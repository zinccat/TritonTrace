# From: 13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2097152, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_convolution_mean_mul_tanh_0', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1968624
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x2 = (xindex // 123039)
    x5 = xindex % 123039
    x0 = xindex % 3969
    x4 = (xindex // 3969)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x5 + (123039*r3) + (1968624*x2)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, 1])
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 16.0
    tmp8 = tmp6 / tmp7
    tmp11 = tmp8 + tmp10
    tmp12 = tmp11 - tmp11
    tmp13 = tl_math.exp(tmp12)
    tmp14 = tmp13 / tmp13
    tmp15 = libdevice.tanh(tmp14)
    tmp16 = 2.0
    tmp17 = tmp15 * tmp16
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0 + (4000*x4)), tmp14, xmask)
    tl.store(out_ptr0 + (x6), tmp17, xmask)
