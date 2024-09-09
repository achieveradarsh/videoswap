# run.py

import os
from face_swapper import download_model, process_video_with_source_image

def main():
    # User input for image and video paths
    source_img_path = input("Enter the path for the source image: ")
    target_video_path = input("Enter the path for the target video: ")
    output_video_path = input("Enter the path for saving the output video: ")

    # Download model from Hugging Face if not already downloaded
    model_url = "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128_fp16.onnx"
    model_path = download_model(model_url, "./checkpoints/inswapper_128.onnx", force_download=True)

    # Process the video with the source image
    process_video_with_source_image(source_img_path, target_video_path, model_path, output_video_path)

if __name__ == "__main__":
    main()
