# From: 60_ConvTranspose3d_Swish_GroupNorm_HardSwish

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'ks3': 'i32', 'ks4': 'i32', 'ks5': 'i32', 'ks6': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks0)
    x2 = ((xindex // ks1) % ks2)
    x5 = xindex // ks3
    x8 = xindex // ks6
    x3 = ((xindex // ks3) % 16)
    x9 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + ((-1)*x5) + ((-1)*((((x0 + x2 + ((-1)*x1) + ((-4)*ks5*x2) + 2*ks5*x1 + 4*x2*ks5*ks5) // ((-1) + 2*ks5)) % ((-1) + 2*ks5)))) + ((-4)*ks5*((((x0 + x2 + ((-1)*x1) + ((-4)*ks5*x2) + 2*ks5*x1 + 4*x2*ks5*ks5) // (1 + ((-4)*ks5) + 4*ks5*ks5)) % ((-1) + 2*ks4)))) + ((-4)*x5*ks5*ks5) + 2*ks4*x5 + 2*ks5*((((x0 + x2 + ((-1)*x1) + ((-4)*ks5*x2) + 2*ks5*x1 + 4*x2*ks5*ks5) // ((-1) + 2*ks5)) % ((-1) + 2*ks5))) + 4*ks5*x5 + 4*ks5*ks5*((((x0 + x2 + ((-1)*x1) + ((-4)*ks5*x2) + 2*ks5*x1 + 4*x2*ks5*ks5) // (1 + ((-4)*ks5) + 4*ks5*ks5)) % ((-1) + 2*ks4))) + ((-8)*ks4*ks5*x5) + 8*ks4*x5*ks5*ks5 + ((((x0 + x2 + ((-1)*x1) + ((-4)*ks5*x2) + 2*ks5*x1 + 4*x2*ks5*ks5) // (1 + ((-4)*ks5) + 4*ks5*ks5)) % ((-1) + 2*ks4)))), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (x0 + ((-1)*x5) + ((-1)*((((x0 + x2 + ((-1)*x1) + ((-4)*ks5*x2) + 2*ks5*x1 + 4*x2*ks5*ks5) // ks0) % ks0))) + ((-4)*ks5*((((x0 + x2 + ((-1)*x1) + ((-4)*ks5*x2) + 2*ks5*x1 + 4*x2*ks5*ks5) // (1 + ((-4)*ks5) + 4*ks5*ks5)) % ks2))) + ((-4)*x5*ks5*ks5) + 2*ks4*x5 + 2*ks5*((((x0 + x2 + ((-1)*x1) + ((-4)*ks5*x2) + 2*ks5*x1 + 4*x2*ks5*ks5) // ks0) % ks0)) + 4*ks5*x5 + 4*ks5*ks5*((((x0 + x2 + ((-1)*x1) + ((-4)*ks5*x2) + 2*ks5*x1 + 4*x2*ks5*ks5) // (1 + ((-4)*ks5) + 4*ks5*ks5)) % ks2)) + ((-8)*ks4*ks5*x5) + 8*ks4*x5*ks5*ks5 + ((((x0 + x2 + ((-1)*x1) + ((-4)*ks5*x2) + 2*ks5*x1 + 4*x2*ks5*ks5) // (1 + ((-4)*ks5) + 4*ks5*ks5)) % ks2))), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x8 // 4), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x8 // 4), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 - tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 + tmp10
    tl.store(out_ptr0 + (x9), tmp11, xmask)
