import os
import cv2
import numpy as np
from PIL import Image
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
        faces = face_analyser.get(frame)
        return sorted(faces, key=lambda x: x.bbox[0])
    except IndexError:
        return None

def swap_face(face_swapper, source_faces, target_faces, source_index, target_index, temp_frame):
    source_face = source_faces[source_index]
    target_face = target_faces[target_index]
    return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)

def process_image_with_source_image(source_img_path: str, target_img_path: str, model: str, output_image_path: str):
    providers = onnxruntime.get_available_providers()
    face_analyser = getFaceAnalyser(model, providers)
    face_swapper = getFaceSwapModel(model)

    # Load and prepare source image
    source_img = Image.open(source_img_path).convert("RGB")
    source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
    source_faces = get_many_faces(face_analyser, source_img)

    if not source_faces:
        raise Exception("No faces found in source image!")

    # Load and prepare target image
    target_img = Image.open(target_img_path).convert("RGB")
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    target_faces = get_many_faces(face_analyser, target_img)

    if not target_faces:
        raise Exception("No faces found in target image!")

    result_img = target_img.copy()
    for i in range(len(target_faces)):
        result_img = swap_face(face_swapper, source_faces, target_faces, 0, i, result_img)

    # Save result
    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    Image.fromarray(result_rgb).save(output_image_path)
    print(f"Result image saved successfully: {output_image_path}")
