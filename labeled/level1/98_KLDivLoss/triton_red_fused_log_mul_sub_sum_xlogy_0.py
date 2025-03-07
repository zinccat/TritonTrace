# From: 98_KLDivLoss

import triton
import triton.language as tl


@triton.jit
def triton_red_fused_log_mul_sub_sum_xlogy_0(input_ptr0, input_ptr1, output_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 64
    rnumel = 8192
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_index
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r1 = r_index
        input_val0 = tl.load(input_ptr0 + (r1 + (8192 * x0)), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        input_val1 = tl.load(input_ptr1 + (r1 + (8192 * x0)), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        
        is_nan0 = tl.extra.cuda.libdevice.isnan(input_val0).to(tl.int1)
        zero_val = 0.0
        is_zero0 = input_val0 == zero_val
        log_input0 = tl.math.log(input_val0)
        input_val0_log_mul = input_val0 * log_input0
        log_mul_result = tl.where(is_zero0, zero_val, input_val0_log_mul)
        nan_val = float("nan")
        log_mul_result_with_nan = tl.where(is_nan0, nan_val, log_mul_result)
        
        log_input1 = tl.math.log(input_val1)
        input_val0_log_input1 = input_val0 * log_input1
        result_diff = log_mul_result_with_nan - input_val0_log_input1
        result_broadcast = tl.broadcast_to(result_diff, [XBLOCK, RBLOCK])
        
        temp_sum = temp_sum + result_broadcast
        temp_sum = tl.where(r_mask & x_mask, temp_sum, temp_sum)
    
    result_sum = tl.sum(temp_sum, 1)[:, None]
    tl.store(output_ptr0 + (x0), result_sum, x_mask)