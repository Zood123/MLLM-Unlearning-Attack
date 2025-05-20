import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from attack_utils import llava_attacker


import pandas as pd
from utils import flatten_dataset

from datasets import load_dataset
from models.DnCNN.DnCNN import DnCNN
from torchvision import transforms

llava_chatbot_prompt = "USER: <image>\n%s\nASSISTANT: "
def parse_args():

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model_path", default="eval_configs/minigpt4_eval.yaml", help="path to model.")
    parser.add_argument("--model_id", default="eval_configs/minigpt4_eval.yaml", help="model id")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--n_iters", type=int, default=500, help="specify the number of iterations for attack.")
    parser.add_argument('--alpha', type=float, default=1, help="step_size of the attack")
    parser.add_argument('--batch_size', type=int, default=8, help="batch_size")
    parser.add_argument("--save_dir", type=str, default='output',
                        help="save directory")


    args = parser.parse_args()
    return args



# ========================================
#             Model Initialization
# ========================================


print('>>> Initializing Models')

from llava_llama_2.utils import load_model_and_processor
args = parse_args()

print('model = ', args.model_path)

tokenizer, model, vis_processor, model_name = load_model_and_processor(args.model_id,args.model_path)
model.eval()

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)



import csv


dataset = load_dataset("MLLMMU/MLLMU-Bench","Test_Set")['train'] 
id_set = {f"{i:03}" for i in range(51, 101)} # 1-50 are test samples, 51-101 are training samples. 
filtered_dataset = dataset.filter(lambda example: example['ID'] in id_set)

queries = []
q_imgs = []
targets = []


for i,sample in enumerate(filtered_dataset):
    images = sample['images']
    for image in images:
        for item in sample['Mask_Task']:
            
            if item['Type'] == "Image_Textual":
                queries.append(item['Question'].replace("__", "[Blank]") + "\nPlease **ONLY** provide the correct answer that should replace the [Blank].") 
                q_imgs.append(image)
                targets.append(item['Ground_Truth'])

            else:
                queries.append(item['Question'].replace("__", "[Blank]") + "\nPlease **ONLY** provide the correct answer that should replace the [Blank].")
                q_imgs.append(image)
                targets.append(item['Ground_Truth'])


# Shuffle while keeping the elements aligned
combined = list(zip(queries, q_imgs, targets))
random.shuffle(combined)
queries, q_imgs, targets = zip(*combined)

# Convert tuples back to lists
queries = list(queries)
q_imgs = list(q_imgs)
targets = list(targets)


my_attacker = llava_attacker.Attacker(args, tokenizer,model,vis_processor,q_imgs,queries, targets, device=model.device, is_rtp=False)

# 336
template_img = 'adversarial_images/prompt_unconstrained_init.bmp'
img = Image.new('RGB', (336 , 336), (0, 0, 0))
img = vis_processor.image_processor(img, return_tensors="pt")["pixel_values"].to(model.device)

text_prompt_template = llava_chatbot_prompt 


adv_img_prompt = my_attacker.attack_unconstrained(text_prompt_template,
                                                            img=img, batch_size = args.batch_size,
                                                            num_iter=args.n_iters, alpha=args.alpha/255)



np.save(adv_img_prompt, '%s/bad_prompt.npy' % args.save_dir)
print('[Done]')
