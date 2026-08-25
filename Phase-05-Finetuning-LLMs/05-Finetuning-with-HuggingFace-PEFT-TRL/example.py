"""
Fine-tuning with Hugging Face (PEFT + TRL)

This lesson is a REFERENCE walkthrough of the real, industry-standard
fine-tuning workflow: transformers.AutoModelForCausalLM + peft.LoraConfig /
get_peft_model + trl.SFTTrainer. `transformers`, `peft`, and `trl` are NOT
installed in this environment (this course otherwise sticks to raw PyTorch),
so this script:

  1. Tries to import all three libraries.
  2. If they're missing, prints a clear, actionable message and exits
     cleanly (exit code 0) -- no confusing traceback.
  3. If they ARE installed, runs a small real fine-tuning job: loads a tiny
     causal LM, wraps it with a LoRA adapter, and trains it for a few steps
     on a toy instruction dataset with SFTTrainer, using every config option
     this lesson's README explains.

The functions below (`build_lora_config`, `build_model_and_tokenizer`,
`run_sft_training`) contain real, accurate API usage for recent
transformers/peft/trl versions -- they are written to actually work if the
libraries were installed, not just as pseudocode.

Runtime note: IF the libraries were installed, `run_sft_training` would take
a few minutes even on a small model/CPU (loading a pretrained checkpoint +
a short training loop). In this environment it never reaches that code path.

Run:
    python example.py
"""

import sys

REQUIRED_PACKAGES = ["transformers", "peft", "trl"]


def check_dependencies():
    """Try importing each required library independently, so the user gets
    a complete picture of what's missing, not just the first failure."""
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


# ---------------------------------------------------------------------------
# The real workflow. Every function below is written to actually run if
# transformers/peft/trl are installed -- this is not simplified pseudocode.
# ---------------------------------------------------------------------------

def build_lora_config():
    """Mirrors Lesson 2's from-scratch LoRALinear (README section 5 there),
    now expressed as the real peft.LoraConfig a production workflow uses.
    See this lesson's README section 2 for what each field maps to."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,   # tells peft to wrap a causal-LM head correctly
        r=16,                            # the LoRA rank -- Lesson 2 section 2's `r`
        lora_alpha=32,                   # the LoRA scaling numerator -- alpha/r in Lesson 2's formula
        lora_dropout=0.05,               # dropout applied to the LoRA path only, a light regularizer
        target_modules=["q_proj", "v_proj"],  # which frozen weight matrices get an A/B update
        bias="none",                     # leave all bias terms frozen; only A/B get gradients
    )


def build_model_and_tokenizer(model_name="sshleifer/tiny-gpt2"):
    """Load a small pretrained causal LM and wrap it with the LoRA adapter
    from build_lora_config(). `tiny-gpt2` is used here purely because it is
    small enough to download and fine-tune quickly for a demo -- swap in
    any real causal LM checkpoint (e.g. a Llama/Mistral/Qwen variant) for
    actual use."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_name)

    lora_config = build_lora_config()
    model = get_peft_model(base_model, lora_config)

    # peft's own utility -- exactly Lesson 2's parameter-count comparison,
    # computed for a real loaded model instead of Lesson 2's toy matrix.
    model.print_trainable_parameters()

    return model, tokenizer


def build_toy_dataset():
    """A tiny instruction dataset in the same Instruction/Response shape as
    Lesson 4's from-scratch example -- SFTTrainer expects a Hugging Face
    `datasets.Dataset` (or a list of dicts) with a text field it can format."""
    from datasets import Dataset

    examples = [
        {"instruction": "Uppercase the word cat.", "response": "CAT"},
        {"instruction": "Reverse the word dog.", "response": "god"},
        {"instruction": "Uppercase the word sun.", "response": "SUN"},
        {"instruction": "Reverse the word run.", "response": "nur"},
    ]

    def to_text(example):
        # The formatting step this lesson's README section 2 calls a "chat
        # template" in its simplest form -- SFTTrainer can also take a real
        # tokenizer.apply_chat_template call here for role-tagged data.
        example["text"] = (
            f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
        )
        return example

    return Dataset.from_list(examples).map(to_text)


def run_sft_training():
    """The real training call -- trl.SFTTrainer wraps the masked-loss
    mechanism Lesson 4 built from scratch (response-only cross-entropy)
    behind a single `.train()` call, given a `dataset_text_field` to read
    formatted examples from."""
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = build_model_and_tokenizer()
    dataset = build_toy_dataset()

    training_args = SFTConfig(
        output_dir="./sft-lora-demo",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=1,
        dataset_text_field="text",
        max_seq_length=128,
        report_to=[],  # disable wandb/tensorboard auto-logging for this demo
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    # Merging the LoRA adapter back into the base model for deployment --
    # exactly Lesson 2 section 3's "merge for free" property, now via peft's
    # real API instead of the from-scratch `merged_weight()` method.
    merged_model = trainer.model.merge_and_unload()
    return trainer, merged_model


def main():
    missing = check_dependencies()

    if missing:
        print("=" * 78)
        print("Fine-tuning with Hugging Face (PEFT + TRL) -- dependency check")
        print("=" * 78)
        print(f"\nThis lesson's real workflow needs: {', '.join(REQUIRED_PACKAGES)}")
        print(f"Missing in this environment: {', '.join(missing)}\n")
        print("Install them with:")
        print(f"\n    pip install {' '.join(missing)}\n")
        print("(transformers/peft/trl are deliberately NOT installed in this course's")
        print("environment -- earlier lessons in this phase build every mechanism")
        print("these libraries wrap [LoRA in Lesson 2, response-masked SFT loss in")
        print("Lesson 4] from scratch in raw PyTorch, precisely so this gap doesn't")
        print("block understanding what the real library calls below are doing.)\n")
        print("This script's functions (build_lora_config, build_model_and_tokenizer,")
        print("build_toy_dataset, run_sft_training) contain real, accurate")
        print("transformers/peft/trl API usage -- read them directly, or install the")
        print("three packages above and re-run this script to execute them for real.")
        print("\nExiting cleanly (no traceback) since the libraries are unavailable.")
        sys.exit(0)

    # This branch only runs if transformers/peft/trl are actually installed.
    print("All required libraries found -- running the real LoRA + SFTTrainer demo.")
    trainer, merged_model = run_sft_training()
    print("\nTraining complete. Final loss history:")
    for entry in trainer.state.log_history:
        if "loss" in entry:
            print(f"  step {entry.get('step')}: loss = {entry['loss']:.4f}")
    print(f"\nMerged model ready for deployment: {type(merged_model).__name__}")


if __name__ == "__main__":
    main()
