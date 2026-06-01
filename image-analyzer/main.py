import argparse
import base64
import os
from statistics import mean

from openai import OpenAI
from PIL import Image, ImageStat


def local_image_analysis(path: str) -> str:
    with Image.open(path) as img:
        stat = ImageStat.Stat(img)
        width, height = img.size
        mode = img.mode

        channel_means = [round(v, 2) for v in stat.mean]
        brightness = round(mean(channel_means), 2)

    return (
        f"Image size: {width}x{height}\n"
        f"Color mode: {mode}\n"
        f"Channel mean values: {channel_means}\n"
        f"Estimated overall brightness: {brightness}"
    )


def openai_image_analysis(path: str, model: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this image and provide a short analysis."},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
                ],
            }
        ],
    )
    return response.output_text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an image with local stats or OpenAI vision.")
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--mode", choices=["local", "openai"], default="local")
    parser.add_argument("--model", default="gpt-4.1-mini")
    args = parser.parse_args()

    if args.mode == "local":
        result = local_image_analysis(args.image)
    else:
        result = openai_image_analysis(args.image, args.model)

    print("Image Analysis")
    print(result)


if __name__ == "__main__":
    main()
