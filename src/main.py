import argparse
import torch
from PIL import Image
from transformers import CLIPTokenizer
from load_model import preload_models_from_standard_weights
from generator import generate

def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion implemented from scratch in PyTorch.")
    
    # Основные аргументы
    parser.add_argument("--prompt", type=str, required=True, help="Текст для генерации изображения (можно на русском)")
    parser.add_argument("--negative_prompt", type=str, default="", help="Негативный промпт")
    parser.add_argument("--image", type=str, default=None, help="Путь к картинке для режима Image2Image")
    parser.add_argument("--strength", type=float, default=0.8, help="Уровень шума для Image2Image (0.0 - 1.0)")
    parser.add_argument("--steps", type=int, default=50, help="Количество шагов диффузии")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости")
    parser.add_argument("--output", type=str, default="output.png", help="Имя итогового файла")
    
    # Пути к файлам (по умолчанию ищутся в папке data)
    parser.add_argument("--model_path", type=str, default="./data/v1-5-pruned-emaonly.ckpt", help="Путь к весам SD")
    parser.add_argument("--vocab_path", type=str, default="./data/vocab.json", help="Путь к словарю CLIP")
    parser.add_argument("--merges_path", type=str, default="./data/merges.txt", help="Путь к файлу merges CLIP")
    
    args = parser.parse_args()

    # Определение устройства
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Загрузка токенизатора и моделей
    print("Loading tokenizer and models... (This may take a minute)")
    tokenizer = CLIPTokenizer(args.vocab_path, merges_file=args.merges_path)
    models = preload_models_from_standard_weights(args.model_path, device)
    print("Models loaded successfully!")

    # Подготовка исходного изображения (если есть)
    input_image = None
    if args.image:
        try:
            input_image = Image.open(args.image).convert("RGB")
            print(f"Loaded input image from {args.image}")
        except Exception as e:
            print(f"Error loading image: {e}")
            return

    # Генерация
    output_array = generate(
        models=models,
        tokenizer=tokenizer,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image=input_image,
        strength=args.strength,
        n_inference_steps=args.steps,
        seed=args.seed,
        device=device
    )

    # Сохранение результата
    img = Image.fromarray(output_array)
    img.save(args.output)
    print(f"Image successfully saved to {args.output}")

if __name__ == "__main__":
    main()