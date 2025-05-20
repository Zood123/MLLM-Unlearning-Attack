import torch
from tqdm import tqdm
import random
from torchvision.utils import save_image
import gc
import numpy as np
from llava_llama_2.utils import train_collate_fn_llava
from model.DnCNN.DnCNN import DnCNN
import os



def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images


class Attacker:

    def __init__(self, args,tokenizer, model,vis_processor,q_imgs, queries, targets, device='cuda:0', is_rtp=False):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.is_rtp = is_rtp
        self.queries=queries
        self.targets = targets
        self.q_imgs = q_imgs
        self.vis_processor = vis_processor
        self.num_targets = len(targets)

        self.loss_buffer = []

        # freeze and set to eval model:
        self.model.eval()
        self.model.requires_grad_(False)

        
        denoiser_path = "/models/DnCNN/checkpoint.pth.tar"
        checkpoint = torch.load(denoiser_path,weights_only=True)
        self.denoiser = DnCNN(image_channels=3, depth=17, n_channels=64)
        self.denoiser = torch.nn.DataParallel(self.denoiser)
        torch.backends.cudnn.benchmark = True
        self.denoiser.to("cuda:0")
        self.denoiser.load_state_dict(checkpoint['state_dict'])
        self.denoiser.eval()
        self.denoiser.requires_grad_(False)
        



    def denoise_loss(self, perturbed_images,denoised_images):
        # Extract features
        with torch.no_grad():
            perturbed_outputs = self.model.vision_tower(perturbed_images, output_hidden_states=True)
            denoised_outputs = self.model.vision_tower(denoised_images, output_hidden_states=True)

        layer_idx = self.model.config.vision_feature_layer
        feat_p = perturbed_outputs.hidden_states[layer_idx][:, 1:]
        feat_d = denoised_outputs.hidden_states[layer_idx][:, 1:]

        feat_p = self.model.multi_modal_projector(feat_p)
        feat_d = self.model.multi_modal_projector(feat_d)

        # Average pool over sequence
        feat_p = torch.mean(feat_p, dim=1)
        feat_d = torch.mean(feat_d, dim=1)

        # Normalize embeddings
        feat_p = torch.nn.functional.normalize(feat_p, dim=1)
        feat_d = torch.nn.functional.normalize(feat_d, dim=1)

        # Cosine similarity: higher = more similar, so we take -mean(similarity) as the loss
        cossim = torch.sum(feat_p * feat_d, dim=1)  # shape [B]
        return -torch.mean(cossim)


    def tv_loss_isotropic(self,img, reduction="mean"):
        dy = img[:, :, 1:, :-1] - img[:, :, :-1, :-1]  # shape (1,3,335,335)
        dx = img[:, :, :-1, 1:] - img[:, :, :-1, :-1]  # shape (1,3,335,335)
        grad_mag = torch.sqrt(dx.pow(2) + dy.pow(2))
        tv = grad_mag.mean( )  # sum over H and W
        return tv 
    
    def attack_unconstrained(self, text_prompt, img,
                             batch_size = 8, num_iter=2000, alpha=1/255):

        print('>>> batch_size:', batch_size)        
        adv_noise = denormalize(img).clone().to(self.device)
        adv_noise.requires_grad_(True)
        
        best_loss = 100
        for t in tqdm(range(num_iter + 1)):
            print(t)
            #optimizer.zero_grad()
            adv_noise.requires_grad_(True)
            batch_queries_targets = random.sample(list(zip(self.queries,self.targets,self.q_imgs)), batch_size)
            batch_pre_q_imgs = [denormalize(self.vis_processor.image_processor(t[2], return_tensors="pt")["pixel_values"].to(self.device)).to(self.device) for t in batch_queries_targets]
            batch_q_imgs = [(t + adv_noise) for t in batch_pre_q_imgs]  
            batch_q_imgs = torch.cat(batch_q_imgs, dim=0)
            
            denoised_batch_pixels = self.denoiser(batch_q_imgs).clamp(0, 1)
            
            normalized_batch_q_imgs =normalize(batch_q_imgs)
            normalized_denoised_imgs = normalize(denoised_batch_pixels)

            batch_queries=[t[0] for t in batch_queries_targets]
            batch_targets = [t[1] for t in batch_queries_targets]
            batch_input_ids, batch_attention_mask, batch_pixels, batch_labels  = train_collate_fn_llava(self.vis_processor,self.q_imgs[0],normalized_batch_q_imgs,batch_queries,batch_targets)

            batch_pixels = batch_pixels.to(self.device)
            batch_input_ids = batch_input_ids.to(self.device)
            batch_labels = batch_labels.to(self.device)
            batch_attention_mask = batch_attention_mask.to(self.device)

            

            outputs = self.model(input_ids=batch_input_ids,
                            attention_mask=batch_attention_mask,
                            pixel_values=batch_pixels,
                            labels=batch_labels)
            loss = outputs.loss

            outputs_d = self.model(input_ids=batch_input_ids,
                            attention_mask=batch_attention_mask,
                            pixel_values=normalized_denoised_imgs,
                            labels=batch_labels)

            loss_d = outputs_d.loss
            denoise_loss = self.denoise_loss(batch_q_imgs,denoised_batch_pixels)
            loss_all = loss+ loss_d + 0.7*denoise_loss

            grad = torch.autograd.grad(loss_all, adv_noise)[0]
            
            adv_noise = (adv_noise.detach() - alpha * grad.detach().sign()).clamp(-12/255, 12/255)
            
            print("target_loss: %f" % (
                loss.item())
                  )
            #print(adv_noise)
            

            if t % 20 == 0 or loss.item()<best_loss:
                if loss.item()<best_loss:
                    best_loss = loss.item()

                adv_img_prompt = adv_noise.clone().detach().cpu()  # denormalize(x_adv)
                np.save('%s/bad_prompt_temp_%d.npy' % (self.args.save_dir, t), adv_img_prompt.numpy())
        return adv_img_prompt.numpy()











