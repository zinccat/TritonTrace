# From: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool3d_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6912000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 15
    x1 = (xindex // 15) % 15
    x2 = (xindex // 225) % 15
    x3 = (xindex // 3375)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (62*x1) + (1922*x2) + (29792*x3)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (62*x1) + (1922*x2) + (29792*x3)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (31 + (2*x0) + (62*x1) + (1922*x2) + (29792*x3)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (32 + (2*x0) + (62*x1) + (1922*x2) + (29792*x3)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (961 + (2*x0) + (62*x1) + (1922*x2) + (29792*x3)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (962 + (2*x0) + (62*x1) + (1922*x2) + (29792*x3)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (992 + (2*x0) + (62*x1) + (1922*x2) + (29792*x3)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (993 + (2*x0) + (62*x1) + (1922*x2) + (29792*x3)), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp15 = 0.125
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (x4), tmp16, None)
