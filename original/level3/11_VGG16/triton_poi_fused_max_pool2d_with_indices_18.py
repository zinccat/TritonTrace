# From: 11_VGG16

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i8', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_18(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 490
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 7)
    y1 = yindex // 7
    y5 = yindex
    y4 = yindex // 49
    y6 = (yindex % 49)
    tmp0 = tl.load(in_ptr0 + (x2 + 1024*y0 + 14336*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (512 + x2 + 1024*y0 + 14336*y1), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (7168 + x2 + 1024*y0 + 14336*y1), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (7680 + x2 + 1024*y0 + 14336*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1, 1], 1, tl.int8)
    tmp4 = tl.full([1, 1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1, 1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1, 1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(out_ptr0 + (x2 + 512*y5), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (y6 + 49*x2 + 25088*y4), tmp16, xmask & ymask)
