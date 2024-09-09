# face_swapper.py

import os
import cv2
import copy
import numpy as np
from PIL import Image
from typing import List, Union
import insightface
import onnxruntime
import requests

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

def process_video_with_source_image(source_img_path: str, target_video_path: str, model: str, output_video_path: str):
    providers = onnxruntime.get_available_providers()
    face_analyser = getFaceAnalyser(model, providers)
    face_swapper = getFaceSwapModel(model)
    
    # Load source image and convert to RGB
    source_img = Image.open(source_img_path).convert("RGB")
    source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
    source_faces = get_many_faces(face_analyser, source_img)
    
    if source_faces is None:
        raise Exception("No faces found in source image!")

    cap = cv2.VideoCapture(target_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        target_faces = get_many_faces(face_analyser, frame)
        num_target_faces = len(target_faces)
        
        if num_target_faces > 0:
            temp_frame = copy.deepcopy(frame)
            for i in range(num_target_faces):
                temp_frame = swap_face(face_swapper, source_faces, target_faces, 0, i, temp_frame)
            
            out.write(temp_frame)
        else:
            out.write(frame)
    
    cap.release()
    out.release()
    print(f'Result video saved successfully: {output_video_path}')
