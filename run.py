# run.py

from face_swapper import download_model, process_video_with_source_image
from google.colab import files

def main():
    # Upload source image
    print("Upload the source image file")
    uploaded = files.upload()
    source_img_path = list(uploaded.keys())[0]
    
    # Upload target video
    print("Upload the target video file")
    uploaded = files.upload()
    target_video_path = list(uploaded.keys())[0]

    # Define output video path
    output_video_path = "/content/result_video.mp4"

    # Download model from Hugging Face if not already downloaded
    model_url = "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128_fp16.onnx"
    model_path = download_model(model_url, "./checkpoints/inswapper_128.onnx", force_download=True)

    # Process the video with the source image
    process_video_with_source_image(source_img_path, target_video_path, model_path, output_video_path)
    
    print(f'Video processing complete. The result video is saved at: {output_video_path}')

if __name__ == "__main__":
    main()
