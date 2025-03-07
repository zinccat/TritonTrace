# From: 95_CrossEntropyLoss

import triton
import triton.language as tl


@triton.jit
def triton_red_fused_nll_loss_forward_1(in_out_ptr0, input_logits_ptr, input_targets_ptr, input_weights_ptr, input_indices_ptr, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    x_indices = xoffset + tl.arange(0, XBLOCK)[:, None]
    rmask_full = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    sum_losses = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    count_nonzero = tl.full([XBLOCK, RBLOCK], 0, tl.int64)
    
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r_indices = rindex
        logits = tl.load(input_logits_ptr + (r_indices), rmask, eviction_policy='evict_first', other=0.0)
        target_values = tl.load(input_targets_ptr + (r_indices), rmask, eviction_policy='evict_first', other=0.0)
        weights = tl.load(input_indices_ptr + (r_indices), rmask, eviction_policy='evict_first', other=0.0)
        
        invalid_index = tl.full([1, 1], -100, tl.int64)
        valid_mask = logits != invalid_index
        valid_indices = tl.where(valid_mask, logits, tl.full([1, 1], 0, tl.int64))
        offset_indices = valid_indices + tl.full([XBLOCK, RBLOCK], 10, tl.int32)
        adjusted_indices = tl.where(valid_indices < 0, offset_indices, valid_indices)
        
        tl.device_assert(((0 <= adjusted_indices) & (adjusted_indices < 10)) | ~rmask, "index out of bounds: 0 <= adjusted_indices < 10")
        
        log_probs = tl.load(input_indices_ptr + (adjusted_indices + (10 * r_indices)), rmask, eviction_policy='evict_last', other=0.0)
        log_prob_diff = log_probs - target_values
        log_weights = tl.math.log(weights)
        loss = log_prob_diff - log_weights
        neg_loss = -loss
        zero_loss = 0.0
        weighted_loss = tl.where(valid_mask, neg_loss, zero_loss)
        broadcasted_loss = tl.broadcast_to(weighted_loss, [XBLOCK, RBLOCK])
        
        sum_losses = sum_losses + broadcasted_loss
        count_nonzero = count_nonzero + valid_mask.to(tl.int64)
    
    total_loss = tl.sum(sum_losses, 1)[:, None]
    total_count = tl.sum(count_nonzero, 1)[:, None].to(tl.float32)
    average_loss = total_loss / total_count
    
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), average_loss, None)