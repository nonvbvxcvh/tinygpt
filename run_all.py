"""
run_all.py
===========
Master pipeline runner.  Executes every step in order:

  Step 1  — Prepare dataset
  Step 2  — Train BPE tokenizer
  Step 3  — Preprocess text → .bin files
  Step 4  — Train the GPT model
  Step 5  — Generate sample text
  Step 6  — Supervised fine-tuning (optional)

Run with:
  python run_all.py              # Full pipeline
  python run_all.py --skip-sft  # Skip fine-tuning
  python run_all.py --gen-only  # Only generate (model already trained)
"""

import sys
import subprocess
import os
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(script: str, description: str):
    print(f"\n{'='*65}")
    print(f"  Running: {description}")
    print(f"  Script : {script}")
    print(f"{'='*65}\n")
    result = subprocess.run(
        [sys.executable, script],
        cwd = ROOT,
    )
    if result.returncode != 0:
        print(f"\nERROR: {script} exited with code {result.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sft",  action="store_true", help="Skip fine-tuning step")
    parser.add_argument("--gen-only",  action="store_true", help="Only run generation")
    args = parser.parse_args()

    if args.gen_only:
        run(os.path.join("training", "generate.py"), "Generate text from trained model")
        return

    run(os.path.join("data",     "prepare_data.py"),     "Step 1 — Download & prepare dataset")
    run(os.path.join("tokenizer","train_tokenizer.py"),  "Step 2 — Train BPE tokenizer")
    run(os.path.join("training", "preprocess.py"),       "Step 3 — Preprocess text → .bin files")
    run(os.path.join("training", "train.py"),            "Step 4 — Train GPT model")
    run(os.path.join("training", "generate.py"),         "Step 5 — Generate text (pre-trained)")

    if not args.skip_sft:
        run(os.path.join("finetune", "sft.py"),          "Step 6 — Supervised fine-tuning")

    print("\n" + "="*65)
    print("  ✓  Full pipeline complete!")
    print("  To launch the web interface: python app/gradio_app.py")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
