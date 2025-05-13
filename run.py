import os
from face_swapper import process_image_with_source_image, download_model

def main():
    source_img_path = input("Enter the path to the source image: ")
    target_img_path = input("Enter the path to the target image: ")
    output_image_path = "output_image.jpg"

    model_url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    model_path = download_model(model_url, "./checkpoints/inswapper_128.onnx")

    process_image_with_source_image(source_img_path, target_img_path, model_path, output_image_path)

    print(f"Output image saved as: {output_image_path}")

if __name__ == "__main__":
    main()
