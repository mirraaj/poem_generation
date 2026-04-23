import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
import argparse

parser = argparse.ArgumentParser(description="Take a sentence as input")

parser.add_argument(
    "--prompt",
    type=str,
    default="The sun is shining",
    help="Input sentence"
)

args = parser.parse_args()
PROMPT = args.prompt

# Path where your trained model is saved
MODEL_PATH = "./ppo_model_finetuned_model_peft"

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path):
    """
    Load trained PPO + PEFT model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    return model, tokenizer


def generate_poem(model, tokenizer, prompt):
    """
    Generate poem from input prompt
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":
    # Load model
    model, tokenizer = load_model(MODEL_PATH)

    # Input prompt
    prompt = PROMPT

    # Generate poem
    poem = generate_poem(model, tokenizer, prompt)

    print("\n=== Generated Poem ===\n")
    print(poem)