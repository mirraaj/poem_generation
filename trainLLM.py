import torch
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

from read_data import load_data

file_path = "format_data/"

def load_trainable_dataset(path='format_data/'):
    poem_files, _ = load_data(path)
    dataset = Dataset.from_dict({"text": poem_files})
    return dataset

def init_models(model_name = 'gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # GPT2 has no pad token
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager")
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def init_models_peft(model_name = "gpt2"):
    model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # GPT2 has no pad token
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def return_tokenized_data(tokenizer, path='format_data/', max_len=128):
    dataset = load_trainable_dataset(path)
    def tokenize_function(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len
         )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def get_data_collector(tokenizer):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
        )
    return data_collator

def train():
    model, tokenizer = init_models()
    tokenized_dataset = return_tokenized_data(tokenizer, path=file_path, max_len=512 )

    training_args = TrainingArguments(
        output_dir="./poem_model_all_data",
        num_train_epochs=3,
        per_device_train_batch_size=10,
        save_steps=100,
        logging_strategy="steps",
        logging_steps=50,
        )
    data_collector = get_data_collector(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collector
        )
    trainer.train()
    trainer.save_model('./finalLLMmodel')
    tokenizer.save_pretrained('./finalLLMmodel')
    return trainer, tokenizer  

def trainPEFT():
    model, tokenizer = init_models_peft()
    tokenized_dataset = return_tokenized_data(tokenizer, path=file_path, max_len=512 )

    training_args = TrainingArguments(
        output_dir="./poem_model_peft_all_data",
        num_train_epochs=3,
        per_device_train_batch_size=10,
        save_steps=100,
        logging_strategy="steps",
        logging_steps=50,
        )
    data_collector = get_data_collector(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collector
        )
    trainer.train()
    trainer.save_model('./llmLoraModel')
    tokenizer.save_pretrained('./llmLoraModel') 
    return trainer, tokenizer  

if __name__=="__main__":
    trainer, tokenizer = trainPEFT()
