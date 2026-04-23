import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel
import argparse

parser = argparse.ArgumentParser(description="Take a sentence as input")

parser.add_argument(
    "--prompt",
    type=str,
    default="The sun is shining",
    help="Input sentence"
)

args = parser.parse_args()

# -------- CONFIG --------
MODEL_PATH_FULL = "./llmModel"        # for full fine-tuned model
MODEL_PATH_LORA = "./llmLoraModel2"   # for PEFT model
USE_LORA = True                      # set False if using full model

PROMPT = args.prompt
MAX_LENGTH = 100


# -------- LOAD MODEL --------
def load_model(model_path, use_lora=False):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    if use_lora:
        # Load base GPT2 first
        base_model = GPT2LMHeadModel.from_pretrained("gpt2")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = GPT2LMHeadModel.from_pretrained(model_path)

    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    return model, tokenizer


# -------- GENERATE POEM --------
def generate_poem(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    poem = tokenizer.decode(output[0], skip_special_tokens=True)
    return poem


# -------- MAIN --------
if __name__ == "__main__":
    print("Loading model...")
    model_path = MODEL_PATH_LORA if USE_LORA else MODEL_PATH_FULL

    model, tokenizer = load_model(model_path, use_lora=USE_LORA)

    print("\nGenerating poem...\n")
    poem = generate_poem(model, tokenizer, PROMPT, MAX_LENGTH)

    print("Prompt:", PROMPT)
    print("\nGenerated Poem:\n")
    print(poem)