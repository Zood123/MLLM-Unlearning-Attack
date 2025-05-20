import torch
import csv
import pandas as pd
import numpy as np
import os
import json

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    AutoProcessor,
    LlavaForConditionalGeneration,
    Idefics2ForConditionalGeneration,
)
from peft import PeftModel




# processor
def load_model_and_processor(model_id,model_path):
    if model_id.startswith("llava"):
        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
    else:
        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = Idefics2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")

    return tokenizer,model, processor,model_path

def load_model_and_processor_clear(model_id, model_path):
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Make sure model_path is correct and absolute if needed
    
    model = LlavaForConditionalGeneration.from_pretrained(
        "therem/llava-1.5-7b-CLEAR-finetune",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, model_path)
    model = model.merge_and_unload()  # optional: merges LoRA into base weights

    return tokenizer, model, processor, model_path

def train_collate_fn_llava(processor,fix_img,batch_q_imgs,batch_queries,batch_targets):
    images = []
    texts = []
    prompt_texts = []
    

    for i in range(len(batch_queries)):
        images.append(fix_img)
        # Construct prompt with question and answer
        prompt = f"USER: <image>\n{batch_queries[i]}\nASSISTANT: {batch_targets[i]}" # {batch_targets[i]}
        # prompt = (f"USER: \n{question}{adv_p} <image> \nASSISTANT: ")
        texts.append(prompt)
        # Prompt only (for labels masking)
        prompt_only = f"USER: <image>\n{batch_queries[i]}\nASSISTANT: "
        prompt_texts.append(prompt_only)

    if len(texts) == 0 or len(images) == 0:
        raise ValueError("Empty batch. No valid images or text in the examples provided.")

    processor.tokenizer.padding_side = "right"

    # Process the batch
    batch = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # Tokenize prompts only (no answer) to determine masking boundary
    prompt_inputs = processor(
        text=prompt_texts,
        images=images,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    

    labels = batch["input_ids"].clone()

    # Mask labels corresponding to the prompt tokens (with padding)
    for i in range(labels.size(0)):  # for each sample in the batch
        
        prompt_input_ids = prompt_inputs["input_ids"][i]

        # Find prompt length by locating padding (or end-of-sequence)
        # Assuming tokenizer.pad_token_id is used for padding
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id in prompt_input_ids:
            prompt_len = (prompt_input_ids != pad_token_id).sum().item()
        else:
            prompt_len = prompt_input_ids.size(0)

        labels[i, :prompt_len] = -100  # mask the prompt part

        # Also mask padding tokens in labels (to avoid unnecessary loss computation)
        labels[i, batch["input_ids"][i] == processor.tokenizer.pad_token_id] = -100
    
    #labels[labels == processor.tokenizer.pad_token_id] = -100
    #print(batch["attention_mask"][0])
    #print(batch["input_ids"][0])
    #print(labels[0])
    #exit()

    return batch["input_ids"], batch["attention_mask"], batch_q_imgs , labels