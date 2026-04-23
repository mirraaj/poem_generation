import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./ppo_model_finetuned_model_peft"
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model = model.to(device)
    model.eval()

    return model, tokenizer


def generate_poem(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.9,
        top_p=0.95
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    model, tokenizer = load_model()

    prompt = "The sun is shining"
    poem = generate_poem(model, tokenizer, prompt)

    print("\n=== Generated Poem ===\n")
    print(poem)