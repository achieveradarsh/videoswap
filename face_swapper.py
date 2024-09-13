import os
import cv2
import copy
import numpy as np
from PIL import Image
import insightface
import onnxruntime
import requests
import shutil

def download_model(model_url: str, model_path: str, force_download=False):
    if not os.path.exists(model_path) or force_download:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        response = requests.get(model_url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
    return model_path

def getFaceSwapModel(model_path: str):
    model = insightface.model_zoo.get_model(model_path)
    return model

def getFaceAnalyser(model_path: str, providers, det_size=(320, 320)):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", root="./checkpoints", providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser

def get_many_faces(face_analyser, frame: np.ndarray):
    try:
        face = face_analyser.get(frame)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None

def swap_face(face_swapper, source_faces, target_faces, source_index, target_index, temp_frame):
    source_face = source_faces[source_index]
    target_face = target_faces[target_index]
    return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)

def extract_frames(target_video_path: str, temp_dir: str):
    cap = cv2.VideoCapture(target_video_path)
    frame_list = []
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(temp_dir, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_list.append(frame_path)
        count += 1
    
    cap.release()
    return frame_list

def combine_frames_to_video(frame_paths: list, output_video_path: str, fps: float, frame_size: tuple):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        out.write(frame)
    
    out.release()

def process_video_with_source_image(source_img_path: str, target_video_path: str, model: str, output_video_path: str):
    providers = ['CPUExecutionProvider']
    face_analyser = getFaceAnalyser(model, providers)
    face_swapper = getFaceSwapModel(model)
    
    # Create temp directory for storing frames
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)

    # Load source image and convert to RGB
    source_img = Image.open(source_img_path).convert("RGB")
    source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
    source_faces = get_many_faces(face_analyser, source_img)
    
    if source_faces is None:
        raise Exception("No faces found in source image!")

    # Extract frames from video and get their paths
    frame_list = extract_frames(target_video_path, temp_dir)

    # Process each frame and apply face swapping
    processed_frame_list = []
    for frame_path in frame_list:
        frame = cv2.imread(frame_path)
        target_faces = get_many_faces(face_analyser, frame)
        
        if target_faces:
            temp_frame = copy.deepcopy(frame)
            for i in range(len(target_faces)):
                temp_frame = swap_face(face_swapper, source_faces, target_faces, 0, i, temp_frame)
            # Save swapped frame
            cv2.imwrite(frame_path, temp_frame)
        
        processed_frame_list.append(frame_path)

    # Combine frames into video
    cap = cv2.VideoCapture(target_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()

    combine_frames_to_video(processed_frame_list, output_video_path, fps, frame_size)

    # Clean up temp directory
    shutil.rmtree(temp_dir)

    print(f'Result video saved successfully: {output_video_path}')
