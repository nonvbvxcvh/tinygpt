"""
app/gradio_app.py
==================
PURPOSE
-------
A simple Gradio web interface that lets you interact with the trained
TinyGPT model in a browser — no command line required.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS GRADIO?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gradio is a Python library that creates web UIs for ML models in minutes.
You define:
  • Inputs (text box, slider, etc.)
  • A Python function that processes inputs
  • Outputs (text, image, audio, etc.)

Gradio auto-generates the HTML/JS/CSS and serves it at http://localhost:7860

HOW TO RUN:
  python app/gradio_app.py

Then open http://localhost:7860 in your browser.

HOW TO SHARE PUBLICLY (Hugging Face Spaces):
  gradio deploy   (from the command line, requires HF account)
  Or: push to a HF Space repo with app.py

CPU INFERENCE:
  Our tiny model has only ~2M parameters — it runs fast enough on CPU.
  Expect ~0.5 seconds per generation on a modern laptop CPU.

"""

import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def load_model():
    """
    Load the best available checkpoint:
    1. SFT-finetuned model (if it exists)
    2. Pre-trained model
    3. Random weights (for testing without training)
    """
    from model.model    import TinyGPT, GPTConfig
    from tokenizers     import Tokenizer

    tokenizer_path  = os.path.join(PROJECT_ROOT, "tokenizer", "tokenizer.json")
    sft_ckpt_path   = os.path.join(PROJECT_ROOT, "checkpoints", "sft_final.pt")
    best_ckpt_path  = os.path.join(PROJECT_ROOT, "checkpoints", "best.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}\n"
            "Run tokenizer/train_tokenizer.py first."
        )
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Load model
    if os.path.exists(sft_ckpt_path):
        model, _ = TinyGPT.load(sft_ckpt_path, device=device)
        model_name = "SFT fine-tuned model"
    elif os.path.exists(best_ckpt_path):
        model, _ = TinyGPT.load(best_ckpt_path, device=device)
        model_name = "Pre-trained model (best checkpoint)"
    else:
        print("WARNING: No checkpoint found. Using random weights.")
        config     = GPTConfig(vocab_size=tokenizer.get_vocab_size())
        model      = TinyGPT(config).to(device)
        model_name = "Untrained model (random weights — run train.py first!)"

    model.eval()
    return model, tokenizer, device, model_name


def build_app():
    try:
        import gradio as gr
    except ImportError:
        print("ERROR: gradio not installed.  Run: pip install gradio")
        sys.exit(1)

    model, tokenizer, device, model_name = load_model()
    print(f"  Loaded: {model_name}")
    print(f"  Device: {device}")

    def generate_text(prompt: str,
                      max_new_tokens: int,
                      temperature: float,
                      top_k: int) -> str:
        """
        Gradio handler: receives UI inputs, runs generation, returns text.
        """
        if not prompt.strip():
            return "Please enter a prompt."

        # Encode the prompt
        enc = tokenizer.encode(prompt)
        ids = torch.tensor(enc.ids, dtype=torch.long).unsqueeze(0).to(device)

        # Generate
        with torch.no_grad():
            out_ids = model.generate(
                ids,
                max_new_tokens = max_new_tokens,
                temperature    = temperature,
                top_k          = top_k if top_k > 0 else None,
            )

        # Decode and return
        text = tokenizer.decode(out_ids[0].tolist())
        return text

    def qa_generate(question: str) -> str:
        """
        Handler for the Q&A tab — formats the question as an instruction.
        """
        if not question.strip():
            return "Please enter a question."

        prompt = f"### Question: {question}\n### Answer:"
        return generate_text(prompt, max_new_tokens=60,
                             temperature=0.5, top_k=30)

    # ── Build the Gradio interface ─────────────────────────────────────────
    with gr.Blocks(title="TinyGPT", theme=gr.themes.Soft()) as demo:

        gr.Markdown(f"""
# 🤖 TinyGPT — Built From Scratch

A tiny GPT-style language model trained entirely from scratch in PyTorch.

**Model:** {model_name}  
**Architecture:** {model.config.num_layers} transformer blocks · 
{model.config.num_heads} heads · 
{model.config.embed_dim}D embeddings · 
{model.num_params():,} parameters
        """)

        with gr.Tabs():

            # ── TAB 1: Story Generation ────────────────────────────────────
            with gr.TabItem("📖 Story Generation"):
                gr.Markdown("""
Enter a story prompt and the model will continue it.
Try: *"Once upon a time there was"* or *"The little dog found a"*
                """)

                with gr.Row():
                    with gr.Column(scale=3):
                        prompt_box = gr.Textbox(
                            label       = "Prompt",
                            placeholder = "Once upon a time there was",
                            lines       = 3,
                        )
                        generate_btn = gr.Button("✨ Generate", variant="primary")

                    with gr.Column(scale=1):
                        max_tokens_slider = gr.Slider(
                            minimum = 20,
                            maximum = 200,
                            value   = 100,
                            step    = 10,
                            label   = "Max new tokens",
                        )
                        temperature_slider = gr.Slider(
                            minimum = 0.1,
                            maximum = 2.0,
                            value   = 0.8,
                            step    = 0.05,
                            label   = "Temperature (creativity)",
                        )
                        top_k_slider = gr.Slider(
                            minimum = 0,
                            maximum = 200,
                            value   = 50,
                            step    = 5,
                            label   = "Top-k (0 = no limit)",
                        )

                output_box = gr.Textbox(
                    label    = "Generated text",
                    lines    = 8,
                    interactive = False,
                )

                generate_btn.click(
                    fn      = generate_text,
                    inputs  = [prompt_box, max_tokens_slider,
                               temperature_slider, top_k_slider],
                    outputs = output_box,
                )

                gr.Examples(
                    examples = [
                        ["Once upon a time there was a little girl named"],
                        ["The small robot wanted to learn"],
                        ["In the deep forest there lived a"],
                        ["One sunny morning the puppy"],
                    ],
                    inputs = [prompt_box],
                )

            # ── TAB 2: Q&A (SFT model) ──────────────────────────────────
            with gr.TabItem("❓ Q&A (Fine-tuned)"):
                gr.Markdown("""
Ask a simple factual question.
Works best if the SFT model was trained (`finetune/sft.py`).
                """)

                qa_input  = gr.Textbox(
                    label       = "Question",
                    placeholder = "What is the sky's color?",
                )
                qa_btn    = gr.Button("Ask", variant="primary")
                qa_output = gr.Textbox(label="Answer", lines=4)

                qa_btn.click(fn=qa_generate, inputs=qa_input, outputs=qa_output)

                gr.Examples(
                    examples = [
                        ["What is the sky's color?"],
                        ["Where do fish live?"],
                        ["Why do birds sing?"],
                        ["What is rain?"],
                    ],
                    inputs = [qa_input],
                )

            # ── TAB 3: About ─────────────────────────────────────────────
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
## How This Model Works

This is a **GPT-style autoregressive language model** built entirely from
scratch using Python and PyTorch.

### Architecture
- **Token embeddings**: each word-piece mapped to a dense vector
- **Positional embeddings**: inject position information
- **Transformer blocks**: N layers of attention + feedforward
- **Causal self-attention**: every token attends to previous tokens only
- **Language model head**: projects to vocabulary-sized logits

### Training
The model was trained by next-token prediction:
given a sequence of tokens, predict the next one.
Loss function: Cross-Entropy.
Optimizer: AdamW with cosine learning rate decay.

### Generation
Text is generated autoregressively:
1. Feed the prompt to the model
2. Sample the next token from the output distribution
3. Append it to the sequence
4. Repeat

The **temperature** controls randomness:
- Low (0.1–0.5): focused, repetitive
- Medium (0.7–0.9): balanced
- High (1.2+): creative, unpredictable

**Top-k** limits sampling to the k most likely tokens,
preventing the model from picking very unlikely words.
                """)

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = False,   # Set True to get a public Gradio link
        inbrowser   = True,
    )
