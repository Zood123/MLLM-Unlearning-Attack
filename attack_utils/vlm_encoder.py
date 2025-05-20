

class Encoder():
    model_path = None
    def __init__(self, mdpth) -> types.NoneType:
        self.model_path = mdpth
    
    @staticmethod
    def compute_cosine(a_vec:np.ndarray , b_vec:np.ndarray):
        """calculate cosine similarity"""
        norms1 = np.linalg.norm(a_vec, axis=1)
        norms2 = np.linalg.norm(b_vec, axis=1)
        dot_products = np.sum(a_vec * b_vec, axis=1)
        cos_similarities = dot_products / (norms1 * norms2) # ndarray with size=1
        return cos_similarities[0]
    
    @abc.abstractmethod
    def calc_cossim(self,pairs:list[tuple[str,str]]):
        """input list of (query, img path) pairs, 
        output list of cosin similarities"""
        res = []
        for p in pairs:
            text_embed = self.embed_text(p[0])
            img_embed = self.embed_img(p[1])
            cossim = self.compute_cosine(text_embed,img_embed)
            res.append(cossim)
        return res

    @abc.abstractmethod
    def embed_img(self,imgpth)->np.ndarray:
        pass

    @abc.abstractmethod
    def embed_text(self,text)->np.ndarray:
        pass






class LlavaEncoder(Encoder):
    def __init__(self, mdpth,device="cuda:0") -> types.NoneType:
        super().__init__(mdpth)
        self.device=device
        self.model = AutoModelForPreTraining.from_pretrained(
        mdpth, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(mdpth)
        self.imgprocessor = AutoImageProcessor.from_pretrained(mdpth)
    # well well well!!!!  average!
    def embed_img(self, imgpth) -> np.ndarray:
        image = Image.open(imgpth)
        # img embedding
        pixel_value = self.imgprocessor(image, return_tensors="pt").pixel_values.to(self.device)
        image_outputs = self.model.vision_tower(pixel_value, output_hidden_states=True)
        selected_image_feature = image_outputs.hidden_states[self.model.config.vision_feature_layer]
        selected_image_feature = selected_image_feature[:, 1:] # by default
        image_features = self.model.multi_modal_projector(selected_image_feature)
        # calculate average to compress the 2th dimension
        image_features = torch.mean(image_features, dim=1).detach().to("cpu").numpy() # well well well!!!!  average!
        return image_features

    def embed_text(self, text) -> np.ndarray:
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        input_embeds = self.model.get_input_embeddings()(input_ids)
        # calculate average to get shape[1, 4096]
        input_embeds = torch.mean(input_embeds, dim=1).detach().to("cpu").numpy()
        return input_embeds




class QwenEncoder(Encoder):
    min_pixels=224*224
    max_pixels=1024*1024

    def __init__(self,mdpth,device="cuda:0"):
        super().__init__(mdpth)
        self.device=device
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(settings["Qwen2_VL_7B"]).to(device)
        self.visual_model = self.model.visual.to(device)
        self.processor = AutoProcessor.from_pretrained(settings["Qwen2_VL_7B"], min_pixels=self.min_pixels, max_pixels=self.max_pixels)
    
    def embed_img(self,imgpth)->np.ndarray:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": imgpth,
                        'max_pixels': self.max_pixels,
                        'min_pixels': self.min_pixels,
                    },
                    {"type": "text", "text": ""},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        pixel_values = inputs["pixel_values"].type(torch.bfloat16)

        image_embeds = self.visual_model(pixel_values, grid_thw=inputs["image_grid_thw"]) # shape: [n tokens, 3584]

        # calculate average to compress the 2th dimension
        image_features = torch.mean(image_embeds, dim=0).detach().to("cpu").numpy().reshape(1,-1)
        return image_features

    def embed_text(self, text)->np.ndarray:
        text = self.processor.apply_chat_template(text, tokenize=False, add_generation_prompt=True)
        input_ids = self.processor.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        input_embeds = self.model.model.embed_tokens(input_ids)
        # calculate average to get shape[1, 3584]
        input_embeds = torch.mean(input_embeds, dim=1).detach().to("cpu").numpy()
        return input_embeds



