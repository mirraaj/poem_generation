from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from datasets import load_dataset, Dataset
import torch
from read_data import load_data_for_rl, load_query_to_poems_dataset
import json
import numpy as np

base_model_name = "gpt2"
adapter_path = "./llmLoraModel"
device = "cuda"
# device = "mps"
# device = "cpu"

def generate_from_peft_llm_model(model, tokenizer):
    prompt = "<POEM>\nI am a god"
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=64,
        do_sample=True,
        temperature=0.9
    )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

def load_peft_model():
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(adapter_path)
    model = model.to(device)

    return model, tokenizer


def reward_function(generated_poem,reference_poems, semantic_model, semantic_tokenizer):

    # reference_poems = topic_poem[topic]['poems']

    def embed(text):
        inputs = semantic_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = semantic_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    scores = []

    gen_emb = embed(generated_poem)

    for ref in reference_poems:
        ref_emb = embed(ref)
        sim = F.cosine_similarity(gen_emb, ref_emb)
        scores.append(sim.item())

    reward = max(scores)
    return reward

def generate_prompt_to_train(topic_poem, policy_model, tokenizer):

    gemerated_poem = {}
    for topic in list(topic_poem.keys()):
        prompt = topic_poem[topic]['prompt']
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        output_ids = policy_model.generate(
            **inputs,
            max_new_tokens=80
        )

        poem = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        topic_poem[topic]['generated'] = poem

    return topic_poem

def make_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    else:
        return obj

def train_PPO_model(prompt_poem, topic_poem, semantic_model, semantic_tokenizer, model, tokenizer, model_path = './llmLoraModel'):
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # tokenizer.pad_token = tokenizer.eos_token

    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
    model = model.to(device)

    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=4,
        mini_batch_size=2,
        gradient_accumulation_steps=1
    )

    ppo_data = []

    for topic in topic_poem.keys():

        prompt = topic_poem[topic]['prompt']
        ppo_data.append({
                "query": prompt,
                "topic" : topic,
                "poems" : topic_poem[topic]['poems']
            })

    ppo_dataset = Dataset.from_list(ppo_data)   
    print(ppo_dataset)

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=ppo_dataset,
        data_collator=lambda data: data
    )
    
    all_stats = []

    for epoch in range(3):
        for i, batch in enumerate(ppo_trainer.dataloader):

            prompts = [item["query"] for item in batch]
            poems = [prompt_poem[prompt] for prompt in prompts]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            # if torch.cuda.is_available():
            inputs = {k: v.to(device) for k, v in inputs.items()}

            input_ids = inputs["input_ids"]   # shape: (B, L)
            query_lengths = [len(q) for q in input_ids]
            query_tensors = [q for q in input_ids]  
            query_tensors = [q for q in query_tensors] 

            # Generate responses
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )

            response_tensors = []

            for j, output in enumerate(outputs):
                response = output[query_lengths[j]:]   # remove prompt part
                response_tensors.append(response)

            # response_tensors = outputs[:, query_tensors.shape[1]:]
            # response_tensors = [r for r in response_tensors]
            responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Compute rewards
            rewards = []
            for res, p in zip(responses, poems):
                reward = reward_function(res, p, semantic_model, semantic_tokenizer)
                rewards.append(reward)

            # Convert to tensors
            # rewards = torch.tensor(rewards).to(device)
            rewards = [torch.tensor(r).to(device) for r in rewards]

            # Run PPO step
            stats = ppo_trainer.step(
                query_tensors,
                response_tensors,
                rewards
            )

            clean_stats = {k: float(v) if torch.is_tensor(v) else v for k, v in stats.items()}
            clean_stats["epoch"] = epoch
            clean_stats["step"] = i

            all_stats.append(clean_stats)
            if i == 2:
                break
        break

    # =========================
    # Save model
    # =========================
    # ppo_trainer.save_pretrained("./ppo_finetuned_model_peft")
    serializable_stats = make_serializable(all_stats)
    with open("ppo_peft_stats.json", "w") as file:
        json.dump(serializable_stats, file, indent=2)
    # df = pd.DataFrame(all_stats)
    # df.to_csv("ppo_peft_stats.csv", index=False)

    return ppo_trainer, tokenizer




if __name__=="__main__":

    model, tokenizer = load_peft_model()
    topic_poem = load_data_for_rl(path = 'format_data/topics/')

    semantic_model = AutoModel.from_pretrained("bert-base-uncased")
    semantic_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    prompt_poem = load_query_to_poems_dataset(path = 'format_data/topics/')
    ppo_trainer, tokenizer = train_PPO_model(prompt_poem, topic_poem, semantic_model, semantic_tokenizer, model, tokenizer, model_path = './llmLoraModel')

    # ppo_trainer.model.pretrained_model.save_pretrained("./ppo_base_model")
    # tokenizer.save_pretrained("./ppo_trainer_finetuned_model_peft")
    # ppo_trainer.save_pretrained("./ppo_trainer_finetuned_model_peft", create_model_card=False)

    model = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model)
    model.save_pretrained("./ppo_model_finetuned_model_peft")

    # Save tokenizer
    tokenizer.save_pretrained("./ppo_model_finetuned_model_peft")

    # Save training state
    torch.save(ppo_trainer.optimizer.state_dict(), "./ppo_model_finetuned_model_peft/optimizer.pt")
    torch.save(ppo_trainer.lr_scheduler.state_dict(), "./ppo_model_finetuned_model_peft/scheduler.pt")

def trainRLmodel():
    model, tokenizer = load_peft_model()
    topic_poem = load_data_for_rl(path = 'format_data/topics/')

    semantic_model = AutoModel.from_pretrained("bert-base-uncased")
    semantic_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    prompt_poem = load_query_to_poems_dataset(path = 'format_data/topics/')
    ppo_trainer, tokenizer = train_PPO_model(prompt_poem, topic_poem, semantic_model, semantic_tokenizer, model, tokenizer, model_path = './llmLoraModel')

    # ppo_trainer.model.pretrained_model.save_pretrained("./ppo_base_model")
    tokenizer.save_pretrained("./ppo_finetuned_model_peft")
    ppo_trainer.save_pretrained("./ppo_finetuned_model_peft", create_model_card=False)