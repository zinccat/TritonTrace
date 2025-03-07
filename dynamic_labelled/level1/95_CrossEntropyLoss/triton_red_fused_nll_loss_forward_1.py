# From: 95_CrossEntropyLoss

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_nll_loss_forward_1red_fused_nll_loss_forward_1(
    output_ptr, input_logits_ptr, input_targets_ptr, input_weights_ptr, num_classes, 
    start_index, num_elements, reduction_size, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    r_base = tl.arange(0, RBLOCK)[None, :]
    sum_losses = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    valid_counts = tl.full([XBLOCK, RBLOCK], 0, tl.int64)

    for r_offset in range(0, reduction_size, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < reduction_size
        r_indices = r_index

        logits = tl.load(input_logits_ptr + (r_indices), r_mask, eviction_policy='evict_first', other=0.0)
        target_indices = tl.load(input_targets_ptr + (r_indices), r_mask, eviction_policy='evict_first', other=0.0)
        weights = tl.load(input_weights_ptr + (r_indices), r_mask, eviction_policy='evict_first', other=0.0)

        invalid_index = tl.full([1, 1], -100, tl.int64)
        valid_mask = logits != invalid_index
        clamped_indices = tl.where(valid_mask, logits, tl.full([1, 1], 0, tl.int64))
        clamped_indices = clamped_indices + num_classes
        clamped_indices = tl.where(clamped_indices < 0, clamped_indices + num_classes, clamped_indices)

        tl.device_assert(((0 <= clamped_indices) & (clamped_indices < num_classes)) | ~r_mask, 
                         "index out of bounds: 0 <= clamped_indices < num_classes")

        log_probs = tl.load(input_logits_ptr + (clamped_indices + num_classes * r_indices), r_mask, eviction_policy='evict_last', other=0.0)
        log_probs = log_probs - target_indices
        log_probs = log_probs - tl.math.log(weights)

        neg_log_probs = -log_probs
        weighted_losses = tl.where(valid_mask, neg_log_probs, tl.full([XBLOCK, RBLOCK], 0.0))
        sum_losses += weighted_losses

        valid_counts += valid_mask.to(tl.int64)

    total_losses = tl.sum(sum_losses, 1)[:, None]
    total_valid_counts = tl.sum(valid_counts, 1)[:, None].to(tl.float32)
    average_losses = total_losses / total_valid_counts

    tl.debug_barrier()
    tl.store(output_ptr + (tl.full([XBLOCK, 1], 0, tl.int32)), average_losses, None)