"""
Based on the lit-llama implementation: https://github.com/Lightning-AI/lit-llama.
Lit-LLaMA is released under the Apache 2.0 license.
"""
import sys
import warnings
from typing import Optional

import torch
warnings.filterwarnings(
    "ignore", 
    message="ComplexHalf support is experimental and many operators don't support it yet"
)

@torch.no_grad()
def generate_from_image(
    model: torch.nn.Module,
    image_ref,
    image_tar,
    image_positions,
    idx: torch.Tensor,
    max_new_tokens: int,
    max_seq_length: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    B, T = idx.shape
    T_new = T + max_new_tokens
    empty = torch.empty(B, T_new, dtype=idx.dtype, device=idx.device)
    empty[:, :T] = idx
    idx = empty

    for t in range(T, T_new):
        idx_cond = idx[:, :t]
        idx_cond = idx_cond if T <= max_seq_length else idx_cond[:, -max_seq_length:]

        # forward
        logits = model(image_ref, image_tar, idx_cond, image_positions)
        logits = logits[:, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # concatenate the new column
        idx[:, t:] = idx_next

    return idx

def truncate_output_to_eos(output, eos_id):
    # TODO: Make this more efficient, terminate generation early
    try:
        eos_pos = output.tolist().index(eos_id)
    except ValueError:
        eos_pos = -1

    output = output[:eos_pos]
    return output