# From: 18_SqueezeNet

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*i8', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32', 5: 'i32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_max_pool2d_with_indices_backward_14', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % ks0
    x1 = (xindex // ks0) % ks0
    x2 = (xindex // ks1)
    x5 = xindex % ks1
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2)))) + ((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2))*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2))) + ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))))) + ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0))))))), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2)))) + ((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2))*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2))) + ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))))) + ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0))))))), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2)))) + ((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2))*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2))) + ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))))) + ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0))))))), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr1 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2)))) + ((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2))*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2))) + ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))))) + ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0))))))), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr0 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2)))) + ((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2))*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2))) + ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0))))))), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr1 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2)))) + ((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2))*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2))) + ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))))) + ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0))))))), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr0 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2)))) + ((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2))*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2))) + ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))))) + ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0))))))), xmask, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr1 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2)))) + ((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2))*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-7) + ks3,  2)),  2)),  2))) + ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))))) + ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0))))))), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1], 3, tl.int32)
    tmp2 = tl.where((tmp0 < 0) != (tmp1 < 0), tl.where(tmp0 % tmp1 != 0, tmp0 // tmp1 - 1, tmp0 // tmp1), tmp0 // tmp1)
    tmp3 = tmp2 * tmp1
    tmp4 = tmp0 - tmp3
    tmp5 = 2*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0))))))
    tmp6 = tmp5 + tmp2
    tmp7 = 2*((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2)))) < (((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0))))))
    tmp8 = tmp7 + tmp4
    tmp9 = ks0
    tmp10 = tmp6 * tmp9
    tmp11 = tmp10 + tmp8
    tmp13 = x5
    tmp14 = tmp11 == tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp14, tmp12, tmp15)
    tmp18 = tl.where((tmp17 < 0) != (tmp1 < 0), tl.where(tmp17 % tmp1 != 0, tmp17 // tmp1 - 1, tmp17 // tmp1), tmp17 // tmp1)
    tmp19 = tmp18 * tmp1
    tmp20 = tmp17 - tmp19
    tmp21 = tmp5 + tmp18
    tmp22 = 2*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0))))))
    tmp23 = tmp22 + tmp20
    tmp24 = tmp21 * tmp9
    tmp25 = tmp24 + tmp23
    tmp27 = tmp25 == tmp13
    tmp28 = ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))
    tmp29 = ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))
    tmp30 = tmp28 < tmp29
    tmp31 = 1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))
    tmp32 = ((ks2) * ((ks2) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks2)))
    tmp33 = tmp31 < tmp32
    tmp34 = tmp30 & tmp33
    tmp35 = tmp34 & tmp27
    tmp36 = tmp16 + tmp26
    tmp37 = tl.where(tmp35, tmp36, tmp16)
    tmp39 = tl.where((tmp38 < 0) != (tmp1 < 0), tl.where(tmp38 % tmp1 != 0, tmp38 // tmp1 - 1, tmp38 // tmp1), tmp38 // tmp1)
    tmp40 = tmp39 * tmp1
    tmp41 = tmp38 - tmp40
    tmp42 = 2*((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) * ((1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) < (1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0))))))
    tmp43 = tmp42 + tmp39
    tmp44 = tmp7 + tmp41
    tmp45 = tmp43 * tmp9
    tmp46 = tmp45 + tmp44
    tmp48 = tmp46 == tmp13
    tmp49 = 1 + ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x1,  2))) + (triton_helpers.div_floor_integer((-1) + x1,  2)) * ((triton_helpers.div_floor_integer((-1) + x1,  2)) > (0)))
    tmp50 = tmp49 < tmp29
    tmp51 = ((0) * ((0) >= (triton_helpers.div_floor_integer((-1) + x0,  2))) + (triton_helpers.div_floor_integer((-1) + x0,  2)) * ((triton_helpers.div_floor_integer((-1) + x0,  2)) > (0)))
    tmp52 = tmp51 < tmp32
    tmp53 = tmp50 & tmp52
    tmp54 = tmp53 & tmp48
    tmp55 = tmp37 + tmp47
    tmp56 = tl.where(tmp54, tmp55, tmp37)
    tmp58 = tl.where((tmp57 < 0) != (tmp1 < 0), tl.where(tmp57 % tmp1 != 0, tmp57 // tmp1 - 1, tmp57 // tmp1), tmp57 // tmp1)
    tmp59 = tmp58 * tmp1
    tmp60 = tmp57 - tmp59
    tmp61 = tmp42 + tmp58
    tmp62 = tmp22 + tmp60
    tmp63 = tmp61 * tmp9
    tmp64 = tmp63 + tmp62
    tmp66 = tmp64 == tmp13
    tmp67 = tmp50 & tmp33
    tmp68 = tmp67 & tmp66
    tmp69 = tmp56 + tmp65
    tmp70 = tl.where(tmp68, tmp69, tmp56)
    tl.store(out_ptr0 + (x4), tmp70, xmask)
