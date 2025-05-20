



def generate_denoised_img_DnCNN(imgpth:list[str], save_dir:str, device:int|torch.device, batch_size = 50, **kwargs):
    model_path = "/home/xzz5508/code/Unlearning/Attack/UMK/UMK/models/DnCNN/checkpoint.pth.tar"
    save_dir = ""

    """
    read all img file under given dir, and convert to RGB
    copy the original size image as denoise000,
    then resize to 224*224,
    denoise with DnCNN and save them to save_path
    
    :path: path of dir or file of image(s)
    """
    denoised_pth = [] # return
    toPilImage = transforms.ToPILImage()
    resized_imgs = []
    img_names = []
    imgpth.sort()
    
    resize = transforms.Compose([transforms.Resize([224,224])])
    totensor = transforms.ToTensor()
    for filepath in imgpth:
        img = Image.open(filepath).convert("RGB")
        img = resize(img)
        filename = os.path.split(filepath)[1]
        # save the original image
        savename=os.path.splitext(filename)[0]+"_denoised_000times.jpg"
        img.save(os.path.join(save_dir,savename))
        denoised_pth.append(os.path.join(save_dir,savename))
        # rconvert to tensor
        img = totensor(img)
        img = torch.unsqueeze(img,0).cuda(device) # type: ignore # 填充一维
        resized_imgs.append(img)
        img_names.append(filename)
    
    # load multi DnCNN nets
    denoisers=[]
    for ckpt in model_path:
        logger.debug(f"loading DnCNN from: {ckpt}")
        checkpoint = torch.load(ckpt)
        denoiser = torch.nn.DataParallel(DnCNN(image_channels=3, depth=17, n_channels=64),device_ids=[device])
        torch.backends.cudnn.benchmark = True
        denoiser.load_state_dict(checkpoint['state_dict'])
        denoiser.cuda(device).eval()
        denoisers.append(denoiser)


    # iterations = range(step,cps*step,step)
    b_num = ceil(len(resized_imgs)/batch_size) # how many runs do we need
    for b in tqdm.tqdm(range(b_num),desc="denoise batch"):
        l = b*batch_size
        r = (b+1)*batch_size if b<b_num-1 else len(resized_imgs)
        # denoise for each part between l and r
        part = resized_imgs[l:r]
        partname = img_names[l:r]
        with torch.no_grad():
            for i,img in enumerate(part):
                for j,d in enumerate(denoisers):
                    outputs = d(img)
                    outputs = torch.clamp(outputs, 0, 1) # remember to clip pixel values
                    denoised = toPilImage(outputs[0].cpu())
                    sn = os.path.splitext(partname[i])[0]+f"_denoised_{j+1}times.jpg"
                    denoised.save(os.path.join(save_dir,sn))
                    denoised_pth.append(os.path.join(save_dir,sn))
    del denoisers
    torch.cuda.empty_cache()
    return denoised_pth



def test():
    denoiser_path = "/home/xzz5508/code/Unlearning/Attack/UMK/UMK/models/DnCNN/checkpoint.pth.tar"
checkpoint = torch.load(denoiser_path)
denoiser = DnCNN(image_channels=3, depth=17, n_channels=64)
denoiser = torch.nn.DataParallel(denoiser)

torch.backends.cudnn.benchmark = True
denoiser.to("cuda")
denoiser.load_state_dict(checkpoint['state_dict'])

save_name = "/home/xzz5508/code/Unlearning/Attack/UMK/UMK/test_images/denoised_img.png"
original_img_path = "/home/xzz5508/code/Unlearning/Attack/UMK/UMK/test_images/original_img.png"
perturbation_path = "/home/xzz5508/code/Unlearning/Attack/UMK/UMK/adv_test_formal_ga_retain/bad_prompt_temp_200.npy"
perturbed_path = "/home/xzz5508/code/Unlearning/Attack/UMK/UMK/test_images/perturbed_img.png"

# Optional resize if needed for consistency (DnCNN may expect 224x224 if trained that way)
transform = transforms.Compose([
    transforms.Resize((336, 336)),  # change if your model uses native res
    transforms.ToTensor()
])
img_tensor = transform(images[0]).unsqueeze(0).to("cuda")  # shape: [1, 3, H, W]
perturb = np.load(perturbation_path)  # shape should be [3, H, W] or [H, W, 3]
# -------- Run Denoiser ----------
with torch.no_grad():
    output = denoiser(img_tensor)
    output = torch.clamp(output, 0, 1)

# -------- Save Output ----------
output_img = transforms.ToPILImage()(output[0].cpu())


perturb_tensor = torch.tensor(perturb, dtype=torch.float32)
# -------- Apply Perturbation --------
perturbed_img = img_tensor.cpu() + perturb_tensor
perturbed_img = torch.clamp(perturbed_img, 0, 1)
perturbed_img = transforms.ToPILImage()(perturbed_img.squeeze(0).cpu())

output_img.save(save_name)
images[0].save(original_img_path)
perturbed_img.save(perturbed_path)

print(f"Denoised image saved")

print(images[0])
exit()