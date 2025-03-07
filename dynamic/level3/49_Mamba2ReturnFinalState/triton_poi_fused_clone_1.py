# From: 49_Mamba2ReturnFinalState

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32', 5: 'i32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x4 = xindex
    y5 = yindex % 128
    y6 = (yindex // 128)
    y1 = (yindex // 16) % 8
    y2 = (yindex // 128) % ks1
    y3 = (yindex // ks2)
    y7 = yindex
    tmp0 = tl.load(in_ptr0 + (y5 + (128*x4) + (128*ks0*y6)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((-1) + ks0 + (y2*(128 // ks1)) + (ks1*y1*(128 // ks1)) + (8*ks1*y3*(128 // ks1))), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x4 + (y2*(128 // ks1)) + (ks1*y1*(128 // ks1)) + (8*ks1*y3*(128 // ks1))), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp4 = tl_math.exp(tmp3)
    tmp5 = tmp0 * tmp4
    tl.store(out_ptr0 + (x4 + (ks0*y7)), tmp5, xmask)
