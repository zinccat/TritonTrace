# From: 95_CrossEntropyLoss

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_nll_loss_forward_1(output_ptr, input_logits_ptr, input_target_ptr, input_weight_ptr, input_ignore_index_ptr, num_classes, num_elements, reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    r_base = tl.arange(0, RBLOCK)[None, :]
    sum_losses = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    count_valid = tl.full([XBLOCK, RBLOCK], 0, tl.int64)

    for r_offset in range(0, reduction_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < reduction_elements
        r_indices = r_index

        logits = tl.load(input_logits_ptr + (r_indices), r_mask, eviction_policy='evict_first', other=0.0)
        target = tl.load(input_target_ptr + (r_indices), r_mask, eviction_policy='evict_first', other=0.0)
        weight = tl.load(input_weight_ptr + (r_indices), r_mask, eviction_policy='evict_first', other=0.0)
        ignore_index = tl.full([1, 1], -100, tl.int64)
        valid_mask = logits != ignore_index
        clamped_target = tl.where(valid_mask, logits, tl.full([1, 1], 0, tl.int64))
        shifted_target = clamped_target + num_classes
        negative_mask = clamped_target < 0
        adjusted_target = tl.where(negative_mask, shifted_target, clamped_target)
        
        tl.device_assert(((0 <= adjusted_target) & (adjusted_target < num_classes)) | ~r_mask, "index out of bounds: 0 <= adjusted_target < num_classes")

        log_probs = tl.load(input_logits_ptr + (adjusted_target + num_classes * r_indices), r_mask, eviction_policy='evict_last', other=0.0)
        log_prob_diff = log_probs - target
        log_weight = tl.math.log(weight)
        loss = log_prob_diff - log_weight
        neg_loss = -loss
        valid_loss = tl.where(valid_mask, neg_loss, tl.full([1, 1], 0.0, tl.float32))
        broadcast_loss = tl.broadcast_to(valid_loss, [XBLOCK, RBLOCK])

        sum_losses += tl.where(r_mask, broadcast_loss, sum_losses)
        count_valid += tl.where(r_mask, valid_mask.to(tl.int64), count_valid)

    total_loss = tl.sum(sum_losses, 1)[:, None]
    total_count = tl.sum(count_valid, 1)[:, None].to(tl.float32)
    average_loss = total_loss / total_count

    tl.debug_barrier()
    tl.store(output_ptr + (tl.full([XBLOCK, 1], 0, tl.int32)), average_loss, None)