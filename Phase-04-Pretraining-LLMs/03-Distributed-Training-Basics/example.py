"""
Distributed Training Basics

Two demos, both runnable on a single CPU (no actual multi-GPU cluster
needed -- the "devices" in part 1 are just independent model copies in
one Python process, standing in for what would be separate accelerators):

  1. A from-scratch SIMULATION of data-parallel training: split a toy
     batch across N simulated devices, compute per-device gradients on
     independent model copies, average them (simulating an all-reduce),
     and verify the result matches the gradient computed directly on
     the full batch on one device.
  2. A memory-footprint calculator comparing full replication vs. each
     ZeRO sharding stage, for a range of model sizes and device counts,
     using the standard 16*Psi-bytes-per-parameter accounting from the
     ZeRO paper (Rajbhandari et al., 2020).

Run:
    python example.py
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. Data-parallel gradient averaging, simulated across N "devices"
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    """A small 2-layer MLP -- standing in for "the model" in this simulation.
    Small enough that computing full-batch vs. sharded gradients by hand is
    fast, large enough to have a real multi-parameter gradient to compare."""

    def __init__(self, d_in=8, d_hidden=16, d_out=1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def compute_gradient(model, x, y):
    """Runs forward+backward on (x, y) with MEAN-reduced MSE loss, returns a
    flat vector of gradients (one long vector, all parameters concatenated)."""
    model.zero_grad()
    pred = model(x)
    loss = F.mse_loss(pred, y)   # mean reduction -- this is what makes shard-averaging exact
    loss.backward()
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()]), loss.item()


def data_parallel_demo():
    print("=" * 74)
    print("1. DATA-PARALLEL GRADIENT AVERAGING, SIMULATED ACROSS N DEVICES")
    print("=" * 74)

    batch_size, num_devices = 32, 4
    d_in = 8
    x = torch.randn(batch_size, d_in)
    y = torch.randn(batch_size, 1)

    base_model = TinyModel(d_in)

    # --- "One giant device": gradient computed directly on the FULL batch ---
    single_device_model = copy.deepcopy(base_model)
    full_batch_grad, full_batch_loss = compute_gradient(single_device_model, x, y)

    # --- N simulated devices: each gets an IDENTICAL model copy (this is what
    # "replicate the model" means in data parallelism) and a DIFFERENT shard
    # of the batch. Each computes its own local gradient completely independently. ---
    shard_size = batch_size // num_devices
    per_device_grads = []
    per_device_losses = []
    for device_id in range(num_devices):
        device_model = copy.deepcopy(base_model)   # same starting weights on every device
        start = device_id * shard_size
        x_shard = x[start:start + shard_size]
        y_shard = y[start:start + shard_size]
        grad, loss_val = compute_gradient(device_model, x_shard, y_shard)
        per_device_grads.append(grad)
        per_device_losses.append(loss_val)
        print(f"  device {device_id}: batch shard = examples [{start}:{start + shard_size}), "
              f"local loss = {loss_val:.4f}")

    # --- All-reduce: average the N local gradients -- this is the ONE
    # communication step data parallelism needs per training step. ---
    all_reduced_grad = torch.stack(per_device_grads).mean(dim=0)

    max_abs_diff = (all_reduced_grad - full_batch_grad).abs().max().item()
    grad_norm = full_batch_grad.norm().item()

    print(f"\nFull-batch gradient norm (one big 'device'):         {grad_norm:.6f}")
    print(f"All-reduced (averaged) gradient norm ({num_devices} devices):  "
          f"{all_reduced_grad.norm().item():.6f}")
    print(f"Max absolute difference between the two gradient vectors: {max_abs_diff:.2e}")
    print(f"\n-> The difference is at floating-point-arithmetic noise level ({max_abs_diff:.1e}),")
    print(f"   not a real discrepancy: because the loss uses MEAN reduction and all")
    print(f"   {num_devices} shards are equal size, averaging {num_devices} independently-computed")
    print(f"   shard gradients is mathematically IDENTICAL to computing the gradient on")
    print(f"   the whole batch at once. This is exactly why data parallelism produces the")
    print(f"   same training trajectory as single-device training on the full batch --")
    print(f"   it just gets there by running the {num_devices} shards' forward/backward passes")
    print(f"   in PARALLEL instead of sequentially, then synchronizing with one all-reduce.")


# ---------------------------------------------------------------------------
# 2. Memory-footprint calculator: full replication vs. ZeRO sharding stages
# ---------------------------------------------------------------------------

# Standard mixed-precision + Adam byte accounting (Rajbhandari et al., 2020,
# "ZeRO", Section 3): per parameter, a data-parallel replica must hold
#   fp16 parameters:      2 bytes
#   fp16 gradients:       2 bytes
#   fp32 master params:   4 bytes  |
#   fp32 Adam momentum:   4 bytes  |-- optimizer state, 12 bytes total
#   fp32 Adam variance:   4 bytes  |
# for a total of 16 bytes/parameter with FULL replication (ZeRO's own "16*Psi" figure).
BYTES_PARAMS_FP16 = 2
BYTES_GRADS_FP16 = 2
BYTES_OPTIMIZER_STATES = 12   # fp32 master weight + momentum + variance


def per_device_bytes(num_params, num_devices, stage):
    """stage: 'baseline' (full replication), 1, 2, or 3 (ZeRO stages)."""
    if stage == "baseline":
        return num_params * (BYTES_PARAMS_FP16 + BYTES_GRADS_FP16 + BYTES_OPTIMIZER_STATES)
    if stage == 1:      # shard optimizer states only
        return num_params * (BYTES_PARAMS_FP16 + BYTES_GRADS_FP16 + BYTES_OPTIMIZER_STATES / num_devices)
    if stage == 2:      # shard optimizer states + gradients
        return num_params * (BYTES_PARAMS_FP16 + (BYTES_GRADS_FP16 + BYTES_OPTIMIZER_STATES) / num_devices)
    if stage == 3:      # shard everything: params, gradients, optimizer states
        return num_params * (BYTES_PARAMS_FP16 + BYTES_GRADS_FP16 + BYTES_OPTIMIZER_STATES) / num_devices
    raise ValueError(stage)


def memory_calculator_demo():
    print("\n" + "=" * 74)
    print("2. PER-DEVICE MEMORY: FULL REPLICATION vs. ZeRO SHARDING STAGES")
    print("=" * 74)

    num_params = 7e9   # a 7B-parameter model -- LLaMA-7B / Mistral-7B scale
    print(f"Model size: {num_params:.0e} parameters, mixed-precision + AdamW "
          f"(16 bytes/param fully replicated)\n")

    device_counts = [1, 2, 4, 8, 16, 32, 64]
    print(f"{'devices':>8}{'baseline (GB)':>16}{'ZeRO-1 (GB)':>14}"
          f"{'ZeRO-2 (GB)':>14}{'ZeRO-3 (GB)':>14}")
    for n in device_counts:
        baseline_gb = per_device_bytes(num_params, n, "baseline") / 1e9
        zero1_gb = per_device_bytes(num_params, n, 1) / 1e9
        zero2_gb = per_device_bytes(num_params, n, 2) / 1e9
        zero3_gb = per_device_bytes(num_params, n, 3) / 1e9
        print(f"{n:>8}{baseline_gb:>16.1f}{zero1_gb:>14.1f}{zero2_gb:>14.1f}{zero3_gb:>14.1f}")

    baseline_1 = per_device_bytes(num_params, 1, "baseline") / 1e9
    zero3_64 = per_device_bytes(num_params, 64, 3) / 1e9
    print(f"\n-> Full replication ('baseline') costs {baseline_1:.1f} GB per device NO MATTER")
    print(f"   how many devices you add -- more devices only help wall-clock time, not")
    print(f"   memory, under plain data parallelism. ZeRO-3, by contrast, drives per-device")
    print(f"   memory down to {zero3_64:.2f} GB at 64 devices -- roughly a {baseline_1 / zero3_64:.0f}x")
    print(f"   reduction -- because every device now stores only its 1/64 SHARD of the")
    print(f"   parameters, gradients, and optimizer states, reconstructing the full values")
    print(f"   on the fly only when a given layer actually needs them for compute.")
    print(f"   Notice ZeRO-1 and ZeRO-2 plateau far above ZeRO-3's curve: they still fully")
    print(f"   replicate parameters (and, for ZeRO-1, gradients too), so the un-sharded")
    print(f"   piece puts a floor under how low per-device memory can go.")


def main():
    data_parallel_demo()
    memory_calculator_demo()


if __name__ == "__main__":
    main()
