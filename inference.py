import os
import argparse
import torch

from resselt import load_from_file
from pepeline import read, save, ImgColor, ImgFormat


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch image upscaling script"
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save results")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    return parser.parse_args()


def load_model(weights_path: str, device: torch.device):
    model = load_from_file(weights_path)
    model = model.to(
        device,
        memory_format=torch.preserve_format,
        non_blocking=True,
    ).eval()
    return model


def process_image(model, img_path: str, device: torch.device):
    img = read(img_path, ImgColor.RGB, ImgFormat.F32).transpose(2, 0, 1)
    img = (
        torch.tensor(img)
        .to(
            device,
            memory_format=torch.preserve_format,
            non_blocking=True,
        )
        .unsqueeze(0)
    )

    with torch.autocast(device.type, torch.float16):
        with torch.inference_mode():
            output = model(img)

    output = output.permute(0, 2, 3, 1).detach().cpu().numpy()[0]
    return output


def main():
    args = parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.weights, device)

    img_list = os.listdir(args.input_dir)
    total = len(img_list)

    for index, img_name in enumerate(img_list, start=1):
        print(
            f"\rProcessing {index}/{total} | {img_name}",
            end="",
            flush=True,
        )

        img_path = os.path.join(args.input_dir, img_name)
        result = process_image(model, img_path, device)

        save(result.copy(), os.path.join(args.output_dir, img_name))

    print("\nDone.")


if __name__ == "__main__":
    main()
