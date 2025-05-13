# run.py

import os
from face_swapper import process_video_with_source_image, download_model

def main():
    # Get file paths from user input
    source_img_path = input("Enter the path to the source image: ")
    target_video_path = input("Enter the path to the target video: ")

    # Define a standard output path
    output_video_path = "output_video.mp4"

    # Define the model URL
    model_url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    model_path = download_model(model_url, "./checkpoints/model.onnx")

    # Run the face swapping process
    process_video_with_source_image(source_img_path, target_video_path, model_path, output_video_path)

    print(f"Output video saved as: {output_video_path}")

if __name__ == "__main__":
    main()
