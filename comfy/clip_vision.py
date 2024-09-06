import mlx.core as mx
import mlx.nn as nn
import os
import json
import logging

import comfy.ops
import comfy.model_patcher
import comfy.model_management
import comfy.clip_model

class Output:
    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, item):
        setattr(self, key, item)

def clip_preprocess(image, size=224):
    mean = mx.array([0.48145466, 0.4578275, 0.40821073], device=image.device, dtype=image.dtype)
    std = mx.array([0.26862954, 0.26130258, 0.27577711], device=image.device, dtype=image.dtype)
    image = mx.moveaxis(image, -1, 1)
    if not (image.shape[2] == size and image.shape[3] == size):
        scale = size / min(image.shape[2], image.shape[3])
        image = mx.nn.functional.interpolate(image, size=(round(scale * image.shape[2]), round(scale * image.shape[3])), mode="bicubic", antialias=True)
        h = (image.shape[2] - size) // 2
        w = (image.shape[3] - size) // 2
        image = image[:, :, h:h+size, w:w+size]
    image = mx.clip((255. * image), 0, 255).round() / 255.0
    return (image - mean.view([3, 1, 1])) / std.view([3, 1, 1])

class ClipVisionModel():
    def __init__(self, json_config):
        with open(json_config) as f:
            config = json.load(f)

        self.image_size = config.get("image_size", 224)
        self.load_device = comfy.model_management.text_encoder_device()
        offload_device = comfy.model_management.text_encoder_offload_device()
        self.dtype = comfy.model_management.text_encoder_dtype(self.load_device)
        self.model = comfy.clip_model.CLIPVisionModelProjection(config, self.dtype, offload_device, comfy.ops.manual_cast)
        self.model.eval()

        self.patcher = comfy.model_patcher.ModelPatcher(self.model, load_device=self.load_device, offload_device=offload_device)

    def load_sd(self, sd):
        return self.model.load_state_dict(sd, strict=False)

    def get_sd(self):
        return self.model.state_dict()

    def encode_image(self, image):
        comfy.model_management.load_model_gpu(self.patcher)
        pixel_values = clip_preprocess(mx.array(image, device=self.load_device), size=self.image_size).astype(mx.float32)
        out = self.model(pixel_values=pixel_values, intermediate_output=-2)

        outputs = Output()
        outputs["last_hidden_state"] = out[0].to(comfy.model_management.intermediate_device())
        outputs["image_embeds"] = out[2].to(comfy.model_management.intermediate_device())
        outputs["penultimate_hidden_states"] = out[1].to(comfy.model_management.intermediate_device())
        return outputs

def convert_to_transformers(sd, prefix):
    sd_k = sd.keys()
    if f"{prefix}transformer.resblocks.0.attn.in_proj_weight" in sd_k:
        keys_to_replace = {
            f"{prefix}class_embedding": "vision_model.embeddings.class_embedding",
            f"{prefix}conv1.weight": "vision_model.embeddings.patch_embedding.weight",
            f"{prefix}positional_embedding": "vision_model.embeddings.position_embedding.weight",
            f"{prefix}ln_post.bias": "vision_model.post_layernorm.bias",
            f"{prefix}ln_post.weight": "vision_model.post_layernorm.weight",
            f"{prefix}ln_pre.bias": "vision_model.pre_layrnorm.bias",
            f"{prefix}ln_pre.weight": "vision_model.pre_layrnorm.weight",
        }

        for x in keys_to_replace:
            if x in sd_k:
                sd[keys_to_replace[x]] = sd.pop(x)

        if f"{prefix}proj" in sd_k:
            sd['visual_projection.weight'] = sd.pop(f"{prefix}proj").transpose(0, 1)

        sd = transformers_convert(sd, prefix, "vision_model.", 48)
    else:
        replace_prefix = {prefix: ""}
        sd = state_dict_prefix_replace(sd, replace_prefix)
    return sd

def load_clipvision_from_sd(sd, prefix="", convert_keys=False):
    if convert_keys:
        sd = convert_to_transformers(sd, prefix)
    if "vision_model.encoder.layers.47.layer_norm1.weight" in sd:
        json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_g.json")
    elif "vision_model.encoder.layers.30.layer_norm1.weight" in sd:
        json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_h.json")
    elif "vision_model.encoder.layers.22.layer_norm1.weight" in sd:
        if sd["vision_model.embeddings.position_embedding.weight"].shape[0] == 577:
            json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_vitl_336.json")
        else:
            json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_vitl.json")
    else:
        return None

    clip = ClipVisionModel(json_config)
    m, u = clip.load_sd(sd)
    if len(m) > 0:
        logging.warning(f"missing clip vision: {m}")
    u = set(u)
    keys = list(sd.keys())
    for k in keys:
        if k not in u:
            t = sd.pop(k)
            del t
    return clip

def load(ckpt_path):
    sd = load_torch_file(ckpt_path)
    if "visual.transformer.resblocks.0.attn.in_proj_weight" in sd:
        return load_clipvision_from_sd(sd, prefix="visual.", convert_keys=True)
    else:
        return load_clipvision_from_sd(sd)
