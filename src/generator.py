import torch
import numpy as np
from tqdm import tqdm
from deep_translator import GoogleTranslator
from ddpm import DDPM

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def translate_to_english(text, source_language='auto'):
    if not text:
        return text
    try:
        translator = GoogleTranslator(source=source_language, target='en')
        return translator.translate(text)
    except Exception as e:
        print(f"Warning: Translation failed ({e}). Using original text.")
        return text

def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def generate(
    models: dict,
    tokenizer,
    prompt: str = "",
    negative_prompt: str = "",
    input_image = None,
    strength: float = 0.8,
    n_inference_steps: int = 50,
    seed: int = 42,
    device: str = "cuda"
):
    with torch.no_grad():
        to_idle = lambda x: x.to(device) if device else x

        print(f"Original prompt: '{prompt}'")
        prompt = translate_to_english(prompt)
        negative_prompt = translate_to_english(negative_prompt)
        print(f"Translated prompt: '{prompt}'")

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)
        
        cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
        cond_context = clip(cond_tokens)
        
        uncond_tokens = tokenizer.batch_encode_plus([negative_prompt], padding="max_length", max_length=77).input_ids
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
        uncond_context = clip(uncond_tokens)

        context = torch.cat([cond_context, uncond_context])
        to_idle(clip)

        sampler = DDPM(generator)
        sampler.set_inference_timesteps(n_inference_steps)

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
        
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            input_image_tensor = input_image_tensor.unsqueeze(0)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            to_idle(encoder)
        else:
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps, desc="Sampling")
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)
            model_input = latents
            model_input = model_input.repeat(2, 1, 1, 1)

            model_output = diffusion(model_input, context, time_embedding)

            output_cond, output_uncond = model_output.chunk(2)
            cfg_scale = 7.5
            model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        
        return images[0]