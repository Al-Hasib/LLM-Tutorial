"""
Mixed Precision and Optimization

Three demos:
  1. AdamW implemented from scratch (the manual update rule, no
     torch.optim) checked against torch.optim.AdamW on the same toy
     problem -- final parameters should match closely.
  2. A warmup + cosine-decay learning-rate schedule, implemented and
     printed as a text table.
  3. A numerical demonstration of fp16 gradient underflow, and how
     loss scaling (scale before backward, unscale after) fixes it.

Run:
    python example.py
"""

import math

import torch
import torch.nn as nn

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. AdamW from scratch vs. torch.optim.AdamW
# ---------------------------------------------------------------------------

def manual_adamw_step(theta, grad, m, v, t, lr, beta1, beta2, eps, weight_decay):
    """One AdamW update, following the README's decoupled-decay formula exactly."""
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad * grad
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    theta = theta - lr * m_hat / (v_hat.sqrt() + eps) - lr * weight_decay * theta
    return theta, m, v


def adamw_from_scratch_demo():
    print("=" * 74)
    print("1. ADAMW FROM SCRATCH vs. torch.optim.AdamW")
    print("=" * 74)

    lr, beta1, beta2, eps, weight_decay = 0.1, 0.9, 0.999, 1e-8, 0.01
    num_steps = 30

    # Toy problem: minimize L(theta) = 0.5 * sum((theta - target)^2), whose
    # gradient is simply (theta - target) -- analytic, so both the manual and
    # torch versions optimize the EXACT same loss surface with no numerical
    # surprises from autograd vs. hand-derived gradients.
    theta_init = torch.randn(5)
    target = torch.randn(5)

    # --- manual AdamW ---
    theta_manual = theta_init.clone()
    m, v = torch.zeros(5), torch.zeros(5)
    for t in range(1, num_steps + 1):
        grad = theta_manual - target
        theta_manual, m, v = manual_adamw_step(
            theta_manual, grad, m, v, t, lr, beta1, beta2, eps, weight_decay
        )

    # --- torch.optim.AdamW, same hyperparameters, same starting point ---
    theta_torch = nn.Parameter(theta_init.clone())
    optimizer = torch.optim.AdamW([theta_torch], lr=lr, betas=(beta1, beta2),
                                   eps=eps, weight_decay=weight_decay)
    for t in range(1, num_steps + 1):
        optimizer.zero_grad()
        loss = 0.5 * ((theta_torch - target) ** 2).sum()
        loss.backward()
        optimizer.step()

    max_diff = (theta_manual - theta_torch.detach()).abs().max().item()
    print(f"Toy problem: minimize 0.5*sum((theta - target)^2), {num_steps} AdamW steps, "
          f"lr={lr}, weight_decay={weight_decay}\n")
    print(f"theta after manual AdamW:        {theta_manual.numpy().round(5)}")
    print(f"theta after torch.optim.AdamW:   {theta_torch.detach().numpy().round(5)}")
    print(f"target (what both are chasing):  {target.numpy().round(5)}")
    print(f"\nMax absolute difference between manual and torch.optim results: {max_diff:.2e}")
    print(f"-> This is floating-point-arithmetic noise, not a real discrepancy: the")
    print(f"   from-scratch update rule (momentum, bias-corrected variance, and")
    print(f"   DECOUPLED weight decay applied directly to theta) reproduces")
    print(f"   torch.optim.AdamW's behavior to {max_diff:.0e} precision.")


# ---------------------------------------------------------------------------
# 2. Warmup + cosine-decay learning-rate schedule
# ---------------------------------------------------------------------------

def warmup_cosine_lr(step, warmup_steps, total_steps, lr_max, lr_min):
    if step < warmup_steps:
        return lr_max * (step / warmup_steps)
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def lr_schedule_demo():
    print("\n" + "=" * 74)
    print("2. WARMUP + COSINE-DECAY LEARNING-RATE SCHEDULE")
    print("=" * 74)

    warmup_steps, total_steps = 20, 200
    lr_max, lr_min = 3e-4, 1e-5
    print(f"warmup_steps={warmup_steps}, total_steps={total_steps}, "
          f"lr_max={lr_max}, lr_min={lr_min}\n")

    checkpoints = [0, 5, 10, 15, 20, 40, 80, 120, 160, 200]
    max_bar_width = 40
    print(f"{'step':>6}{'phase':>10}{'lr':>12}   schedule")
    for step in checkpoints:
        lr = warmup_cosine_lr(min(step, total_steps), warmup_steps, total_steps, lr_max, lr_min)
        phase = "warmup" if step < warmup_steps else "decay"
        bar_len = round(max_bar_width * lr / lr_max)
        bar = "#" * bar_len
        print(f"{step:>6}{phase:>10}{lr:>12.6f}   {bar}")

    print(f"\n-> The learning rate ramps LINEARLY from 0 up to lr_max over the first")
    print(f"   {warmup_steps} steps (early training, when gradients are large and noisy and")
    print(f"   Adam's own moment estimates haven't stabilized yet), then decays smoothly")
    print(f"   toward lr_min following a cosine curve for the rest of training -- fast")
    print(f"   progress early, careful fine settling late.")


# ---------------------------------------------------------------------------
# 3. fp16 gradient underflow, and the loss-scaling fix
# ---------------------------------------------------------------------------

def fp16_underflow_demo():
    print("\n" + "=" * 74)
    print("3. FP16 GRADIENT UNDERFLOW, AND LOSS SCALING AS THE FIX")
    print("=" * 74)

    true_grad = 2e-8   # a small but entirely ordinary gradient value deep in a large network
    scale_factor = 65536.0   # 2^16, a typical loss-scaling constant

    grad_fp32 = torch.tensor(true_grad, dtype=torch.float32)
    grad_cast_directly = grad_fp32.half()   # cast straight to fp16, no scaling

    scaled_grad_fp32 = grad_fp32 * scale_factor
    scaled_grad_fp16 = scaled_grad_fp32.half()          # cast the SCALED value to fp16
    recovered_grad = scaled_grad_fp16.float() / scale_factor   # unscale back to fp32

    print(f"True gradient value (fp32):                     {true_grad:.3e}")
    print(f"Cast directly to fp16 (no scaling):              {grad_cast_directly.item():.3e}")
    print(f"Scaled by {scale_factor:.0f} before casting to fp16:        "
          f"{scaled_grad_fp16.item():.3e}  (the fp16 value actually stored)")
    print(f"Unscaled back to fp32 after the cast:            {recovered_grad.item():.3e}")

    relative_error = abs(recovered_grad.item() - true_grad) / true_grad
    print(f"\n-> Cast directly to fp16, the true gradient underflows to exactly")
    print(f"   {grad_cast_directly.item()} -- that gradient's entire contribution to the")
    print(f"   optimizer step is silently lost. Scaling the value up by {scale_factor:.0f} FIRST")
    print(f"   moves it well inside fp16's representable range before the cast, and")
    print(f"   dividing back out by the same factor afterward recovers the original")
    print(f"   value to within {relative_error:.2%} relative error -- the entire point of")
    print(f"   loss scaling: do the cast where fp16 still has precision to give.")


def main():
    adamw_from_scratch_demo()
    lr_schedule_demo()
    fp16_underflow_demo()


if __name__ == "__main__":
    main()
