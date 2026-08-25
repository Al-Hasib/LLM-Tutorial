"""
Full Fine-tuning vs Parameter-Efficient Fine-tuning

A memory-footprint calculator (plain arithmetic, no model actually
instantiated -- the point is the ORDER OF MAGNITUDE, which real
frameworks confirm exactly). Three demos:

  1. Per-parameter memory cost of full fine-tuning with AdamW
     (weights + gradients + fp32 master copy + Adam's two moments)
     vs. PEFT (frozen base, paid only at 2 bytes/param, plus a tiny
     trainable slice paid at the full training rate).
  2. That comparison swept across realistic model sizes, from 125M to
     70B parameters.
  3. The multi-task deployment angle: storing N full fine-tuned
     checkpoints vs. one shared frozen base plus N tiny adapters.

Run:
    python example.py
"""

# ---------------------------------------------------------------------------
# Constants: the mixed-precision AdamW memory recipe (README section 2)
# ---------------------------------------------------------------------------

BYTES_WEIGHTS_FP16 = 2
BYTES_GRADIENTS_FP16 = 2
BYTES_FP32_MASTER_COPY = 4
BYTES_ADAM_MOMENT_1 = 4
BYTES_ADAM_MOMENT_2 = 4

# Everything a TRAINABLE parameter needs during optimization.
BYTES_PER_TRAINABLE_PARAM = (
    BYTES_WEIGHTS_FP16
    + BYTES_GRADIENTS_FP16
    + BYTES_FP32_MASTER_COPY
    + BYTES_ADAM_MOMENT_1
    + BYTES_ADAM_MOMENT_2
)  # 16 bytes/param

# A FROZEN parameter only ever needs to sit in memory for the forward pass --
# no gradient, no optimizer state, no fp32 master copy.
BYTES_PER_FROZEN_PARAM = BYTES_WEIGHTS_FP16  # 2 bytes/param

GB = 1024 ** 3


def full_finetune_bytes(n_total):
    return n_total * BYTES_PER_TRAINABLE_PARAM


def peft_bytes(n_total, n_trainable):
    frozen = n_total * BYTES_PER_FROZEN_PARAM
    trainable = n_trainable * BYTES_PER_TRAINABLE_PARAM
    return frozen + trainable


# ---------------------------------------------------------------------------
# 1. Per-parameter breakdown, spelled out
# ---------------------------------------------------------------------------

def per_parameter_breakdown_demo():
    print("=" * 78)
    print("1. WHAT ONE TRAINABLE PARAMETER COSTS DURING ADAMW TRAINING")
    print("=" * 78)
    print(f"  weights (fp16)                   {BYTES_WEIGHTS_FP16} bytes")
    print(f"  gradients (fp16)                 {BYTES_GRADIENTS_FP16} bytes")
    print(f"  fp32 master weight copy          {BYTES_FP32_MASTER_COPY} bytes")
    print(f"  Adam moment 1 (fp32)             {BYTES_ADAM_MOMENT_1} bytes")
    print(f"  Adam moment 2 (fp32)             {BYTES_ADAM_MOMENT_2} bytes")
    print(f"  {'-' * 40}")
    print(f"  TOTAL per trainable parameter    {BYTES_PER_TRAINABLE_PARAM} bytes")
    print(f"\n  A FROZEN parameter (PEFT's base model) only pays for inference")
    print(f"  storage: {BYTES_PER_FROZEN_PARAM} bytes/param -- a "
          f"{BYTES_PER_TRAINABLE_PARAM / BYTES_PER_FROZEN_PARAM:.0f}x difference,")
    print("  per parameter, between 'frozen' and 'trainable under AdamW'.")


# ---------------------------------------------------------------------------
# 2. Sweep across realistic model sizes
# ---------------------------------------------------------------------------

MODEL_SIZES = [
    ("GPT-2 small-scale", 125e6),
    ("GPT-2 XL-scale", 1.5e9),
    ("GPT-3-Curie-scale", 6.7e9),
    ("Llama-2-7B-scale", 7e9),
    ("Llama-2-13B-scale", 13e9),
    ("Llama-2-70B-scale", 70e9),
]

# A representative trainable fraction for LoRA-style PEFT (Lesson 2 computes
# the exact figure for a specific config; ~0.1%-1% is the realistic range
# quoted across the LoRA/QLoRA papers for typical rank/target-module choices).
PEFT_TRAINABLE_FRACTION = 0.005  # 0.5% of total parameters trainable


def model_size_sweep_demo():
    print("\n" + "=" * 78)
    print("2. TRAINING MEMORY: FULL FINE-TUNING vs. PEFT, ACROSS MODEL SIZES")
    print("=" * 78)
    print(f"(PEFT assumes a {PEFT_TRAINABLE_FRACTION * 100:.1f}% trainable-parameter "
          f"fraction, in line with typical LoRA configs -- see Lesson 2)\n")

    header = (f"{'model':<20}{'total params':>14}{'full FT (GB)':>15}"
              f"{'PEFT (GB)':>12}{'ratio':>10}")
    print(header)
    for name, n_total in MODEL_SIZES:
        n_trainable = n_total * PEFT_TRAINABLE_FRACTION
        full_gb = full_finetune_bytes(n_total) / GB
        peft_gb = peft_bytes(n_total, n_trainable) / GB
        ratio = full_gb / peft_gb
        print(f"{name:<20}{n_total:>14.2e}{full_gb:>15.1f}{peft_gb:>12.2f}{ratio:>9.1f}x")

    llama7b_full = full_finetune_bytes(7e9) / GB
    llama7b_peft = peft_bytes(7e9, 7e9 * PEFT_TRAINABLE_FRACTION) / GB
    llama7b_inference_only = 7e9 * BYTES_PER_FROZEN_PARAM / GB
    print(f"\n-> Full fine-tuning a 7B model needs about {llama7b_full:.0f} GB -- more than")
    print(f"   an 80 GB A100 has by itself. Just LOADING that same model for inference")
    print(f"   (no training at all) needs only about {llama7b_inference_only:.0f} GB. PEFT training")
    print(f"   sits at about {llama7b_peft:.1f} GB -- barely above the inference-only cost --")
    print(f"   because {(1 - PEFT_TRAINABLE_FRACTION) * 100:.1f}% of the model never needs a gradient,")
    print("   an optimizer state, or an fp32 master copy at all.")

    # The total-memory ratio above tops out around 8x, because BOTH approaches
    # must store the base model's weights just to run a forward pass at all --
    # that shared cost dilutes the gap. The gap that is genuinely multiple
    # orders of magnitude is in the piece that scales with TRAINABLE params
    # only: gradients + the fp32 master copy + Adam's two moments (everything
    # in the 16-byte breakdown except the 2-byte weights themselves).
    overhead_per_trainable_param = BYTES_PER_TRAINABLE_PARAM - BYTES_WEIGHTS_FP16  # 14 bytes
    print(f"\n{'':30s}{'trainable params':>18}{'grad+optimizer overhead (GB)':>30}")
    for name, n_total in MODEL_SIZES:
        n_trainable_full = n_total
        n_trainable_peft = n_total * PEFT_TRAINABLE_FRACTION
        full_overhead_gb = n_trainable_full * overhead_per_trainable_param / GB
        peft_overhead_gb = n_trainable_peft * overhead_per_trainable_param / GB
        print(f"{name + ' (full FT)':<30s}{n_trainable_full:>18.2e}{full_overhead_gb:>30.1f}")
        print(f"{name + ' (PEFT)':<30s}{n_trainable_peft:>18.2e}{peft_overhead_gb:>30.3f}")

    print(f"\n-> Looking only at the gradient + optimizer-state overhead -- the part")
    print(f"   that scales with how many parameters are TRAINABLE, not how many exist --")
    print(f"   the gap is exactly 1/{PEFT_TRAINABLE_FRACTION} = {1 / PEFT_TRAINABLE_FRACTION:.0f}x at every model size, a genuinely")
    print("   multi-order-of-magnitude difference. It just gets partly hidden in the")
    print("   TOTAL-memory figures above because both approaches still have to store")
    print("   the frozen base weights to run a forward pass at all.")


# ---------------------------------------------------------------------------
# 3. Multi-task deployment: N full checkpoints vs. 1 base + N adapters
# ---------------------------------------------------------------------------

def multi_task_storage_demo():
    print("\n" + "=" * 78)
    print("3. STORING FINE-TUNED VARIANTS FOR MULTIPLE TASKS")
    print("=" * 78)

    n_total = 7e9  # Llama-2-7B scale
    n_trainable = n_total * PEFT_TRAINABLE_FRACTION
    num_tasks = 5

    full_checkpoint_gb = n_total * BYTES_WEIGHTS_FP16 / GB       # one fp16 checkpoint
    adapter_checkpoint_gb = n_trainable * BYTES_WEIGHTS_FP16 / GB  # one fp16 adapter

    full_ft_total_gb = num_tasks * full_checkpoint_gb
    peft_total_gb = full_checkpoint_gb + num_tasks * adapter_checkpoint_gb  # 1 shared base

    print(f"Base model: 7B params. Serving {num_tasks} independently fine-tuned tasks.\n")
    print(f"  one full fp16 checkpoint:        {full_checkpoint_gb:.2f} GB")
    print(f"  one fp16 adapter checkpoint:     {adapter_checkpoint_gb * 1024:.1f} MB "
          f"({PEFT_TRAINABLE_FRACTION * 100:.1f}% of the base model's size)\n")
    print(f"  Full fine-tuning: {num_tasks} independent full checkpoints "
          f"= {full_ft_total_gb:.1f} GB total")
    print(f"  PEFT: 1 shared base + {num_tasks} adapters "
          f"= {peft_total_gb:.2f} GB total")
    print(f"\n-> Serving {num_tasks} fine-tuned tasks from one base model needs "
          f"{full_ft_total_gb / peft_total_gb:.1f}x less storage")
    print("   with PEFT than with independently fully fine-tuned copies -- and the gap")
    print("   widens with every additional task, since each new full-FT task adds an")
    print(f"   entire {full_checkpoint_gb:.1f} GB checkpoint, while each new PEFT task adds only")
    print(f"   {adapter_checkpoint_gb * 1024:.1f} MB.")


def main():
    per_parameter_breakdown_demo()
    model_size_sweep_demo()
    multi_task_storage_demo()


if __name__ == "__main__":
    main()
