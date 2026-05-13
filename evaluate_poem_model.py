"""
evaluate_poem_model.py

Evaluation script for GPT-2 + LoRA + PPO poem generation model.

All outputs are saved inside the myoutputs/ folder.

This script computes:
1. Perplexity / validation loss
2. Average semantic reward
3. Diversity: distinct-n
4. Repetition rate
5. Fluency: grammar score if language_tool_python is installed
6. Topic relevance: cosine similarity
7. Creativity / poetic quality:
   - rhyme score
   - imagery score
   - coherence score
   - lexical novelty score
8. Training graphs from trainer_state.json and ppo_peft_stats.json

Recommended installation:

pip install torch transformers peft trl datasets pandas numpy matplotlib scikit-learn tqdm

Optional for grammar score:

pip install language-tool-python

Run example:

python evaluate_poem_model.py \
  --model_path ./ppo_model_finetuned_model_peft \
  --data_path ./format_data/topics \
  --trainer_state_path "./trainer_state(3).json" \
  --ppo_stats_path "./ppo_peft_stats(3).json"

For LoRA-only model:

python evaluate_poem_model.py \
  --model_path ./llmLoraModel2 \
  --data_path ./format_data/topics \
  --trainer_state_path "./trainer_state(3).json" \
  --ppo_stats_path "./ppo_peft_stats(3).json"
"""

import os
import re
import json
import math
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)

try:
    import language_tool_python
    HAS_LANGUAGE_TOOL = True
except Exception:
    HAS_LANGUAGE_TOOL = False


# ---------------------------------------------------------
# Output folder
# ---------------------------------------------------------

OUTPUT_DIR = "myoutputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------

def load_topic_poems(data_path: str):
    """
    Expected folder structure:

    format_data/topics/
        love/
            poem1.txt
            poem2.txt
        nature/
            poem1.txt
            poem2.txt

    Returns:
    topic_poem = {
        "love": {
            "prompt": "Write a poem about love:",
            "poems": [...]
        }
    }
    """

    data_path = Path(data_path)
    topic_poem = {}

    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    for topic_dir in sorted(data_path.iterdir()):
        if not topic_dir.is_dir():
            continue

        topic = topic_dir.name
        poems = []

        for txt_file in sorted(topic_dir.glob("*.txt")):
            text = txt_file.read_text(
                encoding="utf-8",
                errors="ignore"
            ).strip()

            if text:
                poems.append(text)

        if poems:
            topic_poem[topic] = {
                "prompt": f"Write a poem about {topic}:",
                "poems": poems,
            }

    return topic_poem


def flatten_reference_poems(topic_poem):
    """
    Converts topic_poem dictionary into a DataFrame.
    """

    rows = []

    for topic, obj in topic_poem.items():
        for poem in obj["poems"]:
            rows.append({
                "topic": topic,
                "prompt": obj["prompt"],
                "reference_poem": poem,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# 2. Model loading and generation
# ---------------------------------------------------------

def load_generation_model(model_path: str, device: str):
    """
    Loads a saved Hugging Face causal language model.

    This should work for:
    - PPO saved model
    - LoRA saved model if it is saved as a normal HF model
    - GPT-2 style causal LM folders
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()

    return model, tokenizer


@torch.no_grad()
def generate_poems(
    model,
    tokenizer,
    topic_poem,
    device,
    max_new_tokens=100,
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    num_samples_per_topic=3,
):
    """
    Generates poems for each topic prompt.
    """

    generated_rows = []

    for topic, obj in tqdm(topic_poem.items(), desc="Generating poems"):
        prompt = obj["prompt"]

        for sample_id in range(num_samples_per_topic):
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True
            ).to(device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

            full_text = tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            )

            generated_text = full_text

            if full_text.startswith(prompt):
                generated_text = full_text[len(prompt):].strip()

            generated_rows.append({
                "topic": topic,
                "prompt": prompt,
                "sample_id": sample_id,
                "generated_poem": generated_text,
            })

    return pd.DataFrame(generated_rows)


# ---------------------------------------------------------
# 3. Perplexity / validation loss
# ---------------------------------------------------------

@torch.no_grad()
def compute_validation_loss_and_perplexity(
    model,
    tokenizer,
    reference_df,
    device,
    max_length=512,
    batch_size=4,
):
    """
    Computes average causal language-modeling loss
    on the reference poems.

    Perplexity = exp(validation_loss)
    """

    losses = []
    texts = reference_df["reference_poem"].tolist()

    for i in tqdm(
        range(0, len(texts), batch_size),
        desc="Computing validation loss"
    ):
        batch_texts = texts[i:i + batch_size]

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100

        outputs = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=labels,
        )

        losses.append(outputs.loss.item())

    val_loss = float(np.mean(losses))

    if val_loss < 50:
        perplexity = float(math.exp(val_loss))
    else:
        perplexity = float("inf")

    return val_loss, perplexity


# ---------------------------------------------------------
# 4. Semantic model and reward
# ---------------------------------------------------------

def load_semantic_model(device, semantic_model_name="bert-base-uncased"):
    """
    Loads BERT or any Hugging Face encoder model
    for cosine-similarity based semantic evaluation.
    """

    sem_tokenizer = AutoTokenizer.from_pretrained(semantic_model_name)
    sem_model = AutoModel.from_pretrained(semantic_model_name).to(device)
    sem_model.eval()

    return sem_model, sem_tokenizer


@torch.no_grad()
def mean_pool_embedding(
    text,
    sem_model,
    sem_tokenizer,
    device,
    max_length=256
):
    """
    Mean-pools token embeddings using attention mask.
    """

    inputs = sem_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    ).to(device)

    outputs = sem_model(**inputs)

    hidden = outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()

    pooled = torch.sum(hidden * mask, dim=1) / torch.clamp(
        mask.sum(dim=1),
        min=1e-9
    )

    return pooled


@torch.no_grad()
def cosine_sim_text(text_a, text_b, sem_model, sem_tokenizer, device):
    """
    Computes cosine similarity between two texts.
    """

    emb_a = mean_pool_embedding(text_a, sem_model, sem_tokenizer, device)
    emb_b = mean_pool_embedding(text_b, sem_model, sem_tokenizer, device)

    return F.cosine_similarity(emb_a, emb_b).item()


@torch.no_grad()
def compute_semantic_reward_for_poem(
    generated_poem,
    reference_poems,
    sem_model,
    sem_tokenizer,
    device,
):
    """
    This follows your PPO reward idea:

    reward = maximum cosine similarity between generated poem
    and all reference poems of that topic.
    """

    gen_emb = mean_pool_embedding(
        generated_poem,
        sem_model,
        sem_tokenizer,
        device
    )

    scores = []

    for ref in reference_poems:
        ref_emb = mean_pool_embedding(
            ref,
            sem_model,
            sem_tokenizer,
            device
        )

        sim = F.cosine_similarity(gen_emb, ref_emb).item()
        scores.append(sim)

    max_reward = float(max(scores))
    avg_reward = float(np.mean(scores))

    return max_reward, avg_reward


def add_semantic_metrics(
    generated_df,
    topic_poem,
    sem_model,
    sem_tokenizer,
    device
):
    """
    Adds:
    - semantic_reward_max_ref
    - semantic_reward_avg_ref
    - topic_relevance_prompt_cosine
    """

    max_rewards = []
    avg_rewards = []
    topic_prompt_sims = []

    for _, row in tqdm(
        generated_df.iterrows(),
        total=len(generated_df),
        desc="Semantic metrics"
    ):
        topic = row["topic"]
        poem = row["generated_poem"]

        refs = topic_poem[topic]["poems"]
        prompt = topic_poem[topic]["prompt"]

        max_reward, avg_reward = compute_semantic_reward_for_poem(
            poem,
            refs,
            sem_model,
            sem_tokenizer,
            device
        )

        topic_sim = cosine_sim_text(
            poem,
            prompt,
            sem_model,
            sem_tokenizer,
            device
        )

        max_rewards.append(max_reward)
        avg_rewards.append(avg_reward)
        topic_prompt_sims.append(topic_sim)

    generated_df["semantic_reward_max_ref"] = max_rewards
    generated_df["semantic_reward_avg_ref"] = avg_rewards
    generated_df["topic_relevance_prompt_cosine"] = topic_prompt_sims

    return generated_df


# ---------------------------------------------------------
# 5. Diversity and repetition
# ---------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z']+")


def tokenize_words(text):
    """
    Simple word tokenizer.
    """

    return [w.lower() for w in _WORD_RE.findall(str(text))]


def ngrams(tokens, n):
    """
    Returns n-grams from a token list.
    """

    return [
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]


def distinct_n(texts, n=1):
    """
    distinct-n = unique n-grams / total n-grams

    Higher value means more diversity.
    """

    all_ngrams = []

    for text in texts:
        toks = tokenize_words(text)
        all_ngrams.extend(ngrams(toks, n))

    if len(all_ngrams) == 0:
        return 0.0

    return len(set(all_ngrams)) / len(all_ngrams)


def repetition_rate(text, n=3):
    """
    repetition_rate = repeated n-grams / total n-grams

    Lower value means less repetition.
    """

    toks = tokenize_words(text)
    ng = ngrams(toks, n)

    if len(ng) == 0:
        return 0.0

    counts = Counter(ng)
    repeated = sum(c - 1 for c in counts.values() if c > 1)

    return repeated / len(ng)


def add_diversity_metrics(generated_df):
    """
    Adds diversity and repetition metrics.
    """

    generated_df["word_count"] = generated_df["generated_poem"].apply(
        lambda x: len(tokenize_words(x))
    )

    generated_df["repetition_rate_2gram"] = generated_df["generated_poem"].apply(
        lambda x: repetition_rate(x, n=2)
    )

    generated_df["repetition_rate_3gram"] = generated_df["generated_poem"].apply(
        lambda x: repetition_rate(x, n=3)
    )

    generated_df["distinct_1_per_poem"] = generated_df["generated_poem"].apply(
        lambda x: distinct_n([x], n=1)
    )

    generated_df["distinct_2_per_poem"] = generated_df["generated_poem"].apply(
        lambda x: distinct_n([x], n=2)
    )

    return generated_df


# ---------------------------------------------------------
# 6. Fluency / grammar score
# ---------------------------------------------------------

def grammar_error_rate(text, tool=None):
    """
    Uses language_tool_python if installed.

    grammar_error_rate = number of grammar issues / word count

    fluency_score = 1 / (1 + grammar_error_rate)

    Higher fluency_score is better.
    """

    if not HAS_LANGUAGE_TOOL or tool is None:
        return np.nan, np.nan

    words = tokenize_words(text)

    if len(words) == 0:
        return np.nan, np.nan

    matches = tool.check(text)

    error_rate = len(matches) / len(words)
    fluency = 1.0 / (1.0 + error_rate)

    return float(error_rate), float(fluency)


def add_fluency_metrics(generated_df):
    """
    Adds:
    - grammar_error_rate
    - fluency_score_auto
    """

    if HAS_LANGUAGE_TOOL:
        print("Using language_tool_python for grammar/fluency.")
        tool = language_tool_python.LanguageTool("en-US")
    else:
        print("language_tool_python not installed.")
        print("Fluency scores will be NaN.")
        tool = None

    error_rates = []
    fluency_scores = []

    for text in tqdm(
        generated_df["generated_poem"].tolist(),
        desc="Fluency metrics"
    ):
        er, fs = grammar_error_rate(text, tool)
        error_rates.append(er)
        fluency_scores.append(fs)

    generated_df["grammar_error_rate"] = error_rates
    generated_df["fluency_score_auto"] = fluency_scores

    return generated_df


# ---------------------------------------------------------
# 7. Creativity / poetic quality heuristics
# ---------------------------------------------------------

IMAGERY_WORDS = {
    "moon", "sun", "star", "stars", "sky", "night", "dawn", "dusk",
    "river", "sea", "ocean", "rain", "storm", "wind", "flower", "rose",
    "tree", "leaf", "leaves", "bird", "song", "fire", "light", "shadow",
    "dream", "heart", "soul", "silence", "whisper", "gold", "silver",
    "cloud", "mountain", "valley", "garden", "winter", "spring", "autumn",
    "summer", "dark", "bright", "blue", "red", "green", "scent", "tears",
    "dust", "flame", "forest", "meadow", "echo", "mist", "breath",
    "snow", "wave", "waves", "stone", "path", "road", "voice"
}


def get_line_endings(text):
    """
    Gets the final word of each non-empty line.
    """

    lines = [
        line.strip()
        for line in str(text).splitlines()
        if line.strip()
    ]

    endings = []

    for line in lines:
        words = tokenize_words(line)
        if words:
            endings.append(words[-1])

    return endings


def simple_rhyme_key(word):
    """
    Very simple rhyme approximation.

    It takes the substring from the last vowel onward.

    Example:
    night -> ight
    light -> ight
    """

    word = word.lower()
    vowels = "aeiou"

    for i in range(len(word) - 1, -1, -1):
        if word[i] in vowels:
            return word[i:]

    return word[-3:]


def rhyme_score(text):
    """
    Approximate rhyme score based on line endings.

    Higher value means more repeated rhyme endings.
    """

    endings = get_line_endings(text)

    if len(endings) < 2:
        return 0.0

    keys = [simple_rhyme_key(w) for w in endings]
    counts = Counter(keys)

    rhymed_lines = sum(c for c in counts.values() if c >= 2)

    return rhymed_lines / len(keys)


def imagery_score(text):
    """
    Fraction of words that are imagery-related.

    Higher value may indicate more poetic imagery.
    """

    words = tokenize_words(text)

    if not words:
        return 0.0

    imagery_count = sum(1 for w in words if w in IMAGERY_WORDS)

    return imagery_count / len(words)


def lexical_novelty_score(text, reference_texts):
    """
    Measures how many generated words are not present
    in the reference corpus vocabulary.

    Too high may indicate nonsense.
    Too low may indicate copying.
    Use together with semantic reward and fluency.
    """

    gen_words = set(tokenize_words(text))

    ref_counter = Counter()

    for ref in reference_texts:
        ref_counter.update(tokenize_words(ref))

    ref_vocab = set(ref_counter.keys())

    if not gen_words:
        return 0.0

    novel_words = gen_words - ref_vocab

    return len(novel_words) / len(gen_words)


@torch.no_grad()
def coherence_score(text, sem_model, sem_tokenizer, device):
    """
    Computes average cosine similarity between consecutive lines.

    Higher value means lines are semantically more connected.
    """

    lines = [
        line.strip()
        for line in str(text).splitlines()
        if len(line.strip()) > 3
    ]

    if len(lines) < 2:
        return np.nan

    sims = []

    for a, b in zip(lines[:-1], lines[1:]):
        sims.append(
            cosine_sim_text(
                a,
                b,
                sem_model,
                sem_tokenizer,
                device
            )
        )

    return float(np.mean(sims))


def add_poetic_quality_metrics(
    generated_df,
    topic_poem,
    sem_model,
    sem_tokenizer,
    device
):
    """
    Adds:
    - rhyme_score
    - imagery_score
    - lexical_novelty_score
    - coherence_score_linewise
    - poetic_quality_auto
    """

    all_refs = []

    for obj in topic_poem.values():
        all_refs.extend(obj["poems"])

    generated_df["rhyme_score"] = generated_df["generated_poem"].apply(
        rhyme_score
    )

    generated_df["imagery_score"] = generated_df["generated_poem"].apply(
        imagery_score
    )

    generated_df["lexical_novelty_score"] = generated_df["generated_poem"].apply(
        lambda x: lexical_novelty_score(x, all_refs)
    )

    coherence_scores = []

    for text in tqdm(
        generated_df["generated_poem"].tolist(),
        desc="Coherence metrics"
    ):
        coherence_scores.append(
            coherence_score(
                text,
                sem_model,
                sem_tokenizer,
                device
            )
        )

    generated_df["coherence_score_linewise"] = coherence_scores

    generated_df["poetic_quality_auto"] = (
        0.25 * generated_df["rhyme_score"].fillna(0)
        + 0.25 * generated_df["imagery_score"].fillna(0)
        + 0.25 * generated_df["coherence_score_linewise"].fillna(0)
        + 0.25 * generated_df["lexical_novelty_score"].fillna(0)
    )

    return generated_df


# ---------------------------------------------------------
# 8. Human rating sheet
# ---------------------------------------------------------

def create_human_rating_sheet(generated_df, output_path):
    """
    Creates a CSV file for human evaluation.

    Human rating is recommended for:
    - fluency
    - creativity
    - coherence
    - poetic quality
    """

    sheet = generated_df[
        [
            "topic",
            "prompt",
            "sample_id",
            "generated_poem"
        ]
    ].copy()

    sheet["fluency_human_1_to_5"] = ""
    sheet["creativity_human_1_to_5"] = ""
    sheet["coherence_human_1_to_5"] = ""
    sheet["topic_relevance_human_1_to_5"] = ""
    sheet["comments"] = ""

    sheet.to_csv(output_path, index=False)

    return sheet


# ---------------------------------------------------------
# 9. Plot training logs
# ---------------------------------------------------------

def plot_trainer_state(trainer_state_path, output_dir):
    """
    Plots LoRA supervised fine-tuning loss
    and learning rate from trainer_state.json.
    """

    if not trainer_state_path:
        print("No trainer_state_path provided. Skipping LoRA plots.")
        return

    if not os.path.exists(trainer_state_path):
        print(f"trainer_state file not found: {trainer_state_path}")
        print("Skipping LoRA plots.")
        return

    with open(trainer_state_path, "r") as f:
        state = json.load(f)

    logs = state.get("log_history", [])
    rows = [x for x in logs if "loss" in x]

    if not rows:
        print("No loss entries found in trainer_state.json.")
        return

    df = pd.DataFrame(rows)

    df.to_csv(
        os.path.join(output_dir, "lora_training_log.csv"),
        index=False
    )

    plt.figure()
    plt.plot(df["step"], df["loss"])
    plt.xlabel("Training step")
    plt.ylabel("Training loss")
    plt.title("LoRA Fine-tuning Loss")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "lora_loss_curve.png"),
        dpi=200
    )
    plt.close()

    if "learning_rate" in df.columns:
        plt.figure()
        plt.plot(df["step"], df["learning_rate"])
        plt.xlabel("Training step")
        plt.ylabel("Learning rate")
        plt.title("LoRA Learning Rate Schedule")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "lora_learning_rate.png"),
            dpi=200
        )
        plt.close()


def plot_ppo_stats(ppo_stats_path, output_dir):
    """
    Plots PPO reward, loss, KL and entropy curves
    from ppo_peft_stats.json.
    """

    if not ppo_stats_path:
        print("No ppo_stats_path provided. Skipping PPO plots.")
        return

    if not os.path.exists(ppo_stats_path):
        print(f"PPO stats file not found: {ppo_stats_path}")
        print("Skipping PPO plots.")
        return

    with open(ppo_stats_path, "r") as f:
        stats = json.load(f)

    if not isinstance(stats, list):
        print("ppo_peft_stats.json should contain a list of PPO step dictionaries.")
        print("Skipping PPO plots.")
        return

    if len(stats) == 0:
        print("ppo_peft_stats.json is empty.")
        print("Skipping PPO plots.")
        return

    df = pd.DataFrame(stats)

    df.to_csv(
        os.path.join(output_dir, "ppo_training_log.csv"),
        index=False
    )

    if "global_step" not in df.columns:
        df["global_step"] = np.arange(len(df))

    plot_keys = [
        ("ppo/mean_scores", "PPO Mean Reward", "ppo_mean_reward.png"),
        ("ppo/loss/total", "PPO Total Loss", "ppo_total_loss.png"),
        ("ppo/loss/policy", "PPO Policy Loss", "ppo_policy_loss.png"),
        ("ppo/loss/value", "PPO Value Loss", "ppo_value_loss.png"),
        ("objective/kl", "PPO KL Divergence", "ppo_kl.png"),
        ("objective/entropy", "PPO Entropy", "ppo_entropy.png"),
    ]

    for key, title, filename in plot_keys:
        if key in df.columns:
            plt.figure()
            plt.plot(df["global_step"], df[key])
            plt.xlabel("PPO step")
            plt.ylabel(key)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, filename),
                dpi=200
            )
            plt.close()


# ---------------------------------------------------------
# 10. Summary report
# ---------------------------------------------------------

def summarize_metrics(generated_df, val_loss, perplexity, output_dir):
    """
    Saves summary CSV and JSON.
    """

    metric_cols = [
        "semantic_reward_max_ref",
        "semantic_reward_avg_ref",
        "topic_relevance_prompt_cosine",
        "distinct_1_per_poem",
        "distinct_2_per_poem",
        "repetition_rate_2gram",
        "repetition_rate_3gram",
        "grammar_error_rate",
        "fluency_score_auto",
        "rhyme_score",
        "imagery_score",
        "coherence_score_linewise",
        "lexical_novelty_score",
        "poetic_quality_auto",
        "word_count",
    ]

    summary = {
        "validation_loss": val_loss,
        "perplexity": perplexity,
        "corpus_distinct_1": distinct_n(
            generated_df["generated_poem"].tolist(),
            n=1
        ),
        "corpus_distinct_2": distinct_n(
            generated_df["generated_poem"].tolist(),
            n=2
        ),
        "corpus_distinct_3": distinct_n(
            generated_df["generated_poem"].tolist(),
            n=3
        ),
    }

    for col in metric_cols:
        if col in generated_df.columns:
            numeric_col = pd.to_numeric(
                generated_df[col],
                errors="coerce"
            )

            summary[f"{col}_mean"] = float(numeric_col.mean())
            summary[f"{col}_std"] = float(numeric_col.std())

    summary_df = pd.DataFrame([summary])

    summary_df.to_csv(
        os.path.join(output_dir, "evaluation_summary.csv"),
        index=False
    )

    with open(
        os.path.join(output_dir, "evaluation_summary.json"),
        "w"
    ) as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------
# 11. Main function
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="./ppo_model_finetuned_model_peft",
        help="Path to saved PPO or LoRA model."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="./format_data/topics",
        help="Path to topic-wise poem dataset."
    )

    parser.add_argument(
        "--trainer_state_path",
        type=str,
        default="./trainer_state.json",
        help="Path to trainer_state.json for LoRA training plots."
    )

    parser.add_argument(
        "--ppo_stats_path",
        type=str,
        default="./ppo_peft_stats.json",
        help="Path to PPO stats JSON file."
    )

    parser.add_argument(
        "--semantic_model_name",
        type=str,
        default="bert-base-uncased",
        help="Semantic model used for cosine similarity."
    )

    parser.add_argument(
        "--num_samples_per_topic",
        type=int,
        default=3,
        help="Number of poems to generate for each topic."
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum new tokens generated per poem."
    )

    parser.add_argument(
        "--max_eval_length",
        type=int,
        default=512,
        help="Maximum token length for validation loss calculation."
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Batch size for validation loss calculation."
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature."
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling."
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling."
    )

    args = parser.parse_args()

    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Poem Generation Model Evaluation")
    print("=" * 80)
    print(f"Using device: {device}")
    print(f"All outputs will be saved in: {output_dir}")
    print("=" * 80)

    print("\nLoading topic poems...")
    topic_poem = load_topic_poems(args.data_path)

    if not topic_poem:
        raise ValueError(f"No topic poems found in {args.data_path}")

    print(f"Number of topics found: {len(topic_poem)}")

    reference_df = flatten_reference_poems(topic_poem)

    reference_path = os.path.join(output_dir, "reference_poems.csv")
    reference_df.to_csv(reference_path, index=False)

    print(f"Number of reference poems: {len(reference_df)}")
    print(f"Saved reference poems to: {reference_path}")

    print("\nLoading generation model...")
    model, tokenizer = load_generation_model(args.model_path, device)

    print("\nLoading semantic model...")
    sem_model, sem_tokenizer = load_semantic_model(
        device,
        args.semantic_model_name
    )

    print("\nGenerating poems...")
    generated_df = generate_poems(
        model=model,
        tokenizer=tokenizer,
        topic_poem=topic_poem,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_samples_per_topic=args.num_samples_per_topic,
    )

    raw_generation_path = os.path.join(
        output_dir,
        "generated_poems_raw.csv"
    )

    generated_df.to_csv(raw_generation_path, index=False)

    print(f"Saved raw generated poems to: {raw_generation_path}")

    print("\nComputing validation loss and perplexity...")
    val_loss, perplexity = compute_validation_loss_and_perplexity(
        model=model,
        tokenizer=tokenizer,
        reference_df=reference_df,
        device=device,
        max_length=args.max_eval_length,
        batch_size=args.eval_batch_size,
    )

    print(f"Validation loss: {val_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")

    print("\nComputing semantic reward and topic relevance...")
    generated_df = add_semantic_metrics(
        generated_df,
        topic_poem,
        sem_model,
        sem_tokenizer,
        device,
    )

    print("\nComputing diversity and repetition...")
    generated_df = add_diversity_metrics(generated_df)

    print("\nComputing fluency...")
    generated_df = add_fluency_metrics(generated_df)

    print("\nComputing poetic quality metrics...")
    generated_df = add_poetic_quality_metrics(
        generated_df,
        topic_poem,
        sem_model,
        sem_tokenizer,
        device,
    )

    generated_metrics_path = os.path.join(
        output_dir,
        "generated_poems_with_metrics.csv"
    )

    generated_df.to_csv(generated_metrics_path, index=False)

    print(f"Saved generated poems with metrics to: {generated_metrics_path}")

    print("\nCreating human rating sheet...")
    human_sheet_path = os.path.join(
        output_dir,
        "human_rating_sheet.csv"
    )

    create_human_rating_sheet(
        generated_df,
        human_sheet_path
    )

    print(f"Saved human rating sheet to: {human_sheet_path}")

    print("\nPlotting training curves...")
    plot_trainer_state(
        args.trainer_state_path,
        output_dir
    )

    plot_ppo_stats(
        args.ppo_stats_path,
        output_dir
    )

    print("\nWriting final summary...")
    summary = summarize_metrics(
        generated_df,
        val_loss,
        perplexity,
        output_dir
    )

    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)

    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\n" + "=" * 80)
    print("Saved files")
    print("=" * 80)

    saved_files = [
        "reference_poems.csv",
        "generated_poems_raw.csv",
        "generated_poems_with_metrics.csv",
        "evaluation_summary.csv",
        "evaluation_summary.json",
        "human_rating_sheet.csv",
        "lora_training_log.csv",
        "lora_loss_curve.png",
        "lora_learning_rate.png",
        "ppo_training_log.csv",
        "ppo_mean_reward.png",
        "ppo_total_loss.png",
        "ppo_policy_loss.png",
        "ppo_value_loss.png",
        "ppo_kl.png",
        "ppo_entropy.png",
    ]

    for filename in saved_files:
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            print(f"- {path}")

    print("\nAll outputs saved inside:")
    print(output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()