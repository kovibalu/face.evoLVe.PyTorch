import cv2
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

from align.align_trans import get_reference_facial_points
from align.detector import load_detect_faces_models, process_faces
from align.visualization_utils import show_results
from util.extract_feature_v2 import extract_feature_for_img, load_face_id_model

STREAM_DIR = '/home/ec2-user/projects/facelab-data/stream-data'
RESULT_DIR = '/home/ec2-user/projects/facelab-data/results'
ID_FILES = {
    'kovi': '/home/ec2-user/projects/facelab-data/test_Aligned/kovi_final.npy',
    'nee': '/home/ec2-user/projects/facelab-data/test_Aligned/neelam_final.npy',
}


def main():
    # cap = cv2.VideoCapture(0)
    print('Loading models...')
    det_models = load_detect_faces_models()
    face_id_model = load_face_id_model()
    id_npy = load_id_files()
    crop_size = 112
    max_size = 1024
    reference = get_reference_facial_points(default_square=True)

    print('Starting image processing...')
    image_names = list(os.listdir(STREAM_DIR))
    for img_idx in tqdm(range(len(image_names))):
        image_name = image_names[img_idx]
        pil_img = Image.open(os.path.join(STREAM_DIR, image_name))
        if False:
            ret, image_np = cap.read()
            # BGR -> RGB
            pil_img = Image.fromarray(image_np[..., ::-1])

        pil_img.thumbnail((max_size, max_size))
        # Detect bboxes and landmarks for all faces in the image and warp the
        # faces.
        face_results = process_faces(
            img=pil_img,
            det_models=det_models,
            reference=reference,
            crop_size=crop_size)

        identity_list = []
        for fr_idx, face_result in enumerate(face_results):
            features = extract_feature_for_img(
                img=face_result.warped_face,
                backbone=face_id_model)
            #features is tensor, so converting to numpy arr below
            identity = check_identity(
                id_npy=id_npy,
                query_features=features.numpy())
            identity_list.append(identity)

        # Visualize the results
        viz_img = show_results(
            img=pil_img,
            bounding_boxes=[
                fr.bounding_box
                for fr in face_results
            ],
            facial_landmarks=[
                fr.landmark
                for fr in face_results
            ],
            names=identity_list)

        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)
        viz_img.save(os.path.join(RESULT_DIR, '{}-stream.jpg'.format(img_idx)))


def load_id_files():
    id_npy = {}
    for id,path in ID_FILES.items():
        id_npy[id] = np.load(path)
    return id_npy


def check_identity(id_npy,query_features):
    distances_from_id = {}
    for name, id_npy_arr in id_npy.items():
        distances_from_id[name] = []
        for id_npy_row in id_npy_arr:
            dist = np.linalg.norm(id_npy_row - query_features)
            distances_from_id[name].append(dist)

    result = np.finfo(float).max
    name_result = ''
    for name, distances in distances_from_id.items():
        avg = np.mean(distances)
        if avg < result:
            result = avg
            name_result = name

    return name_result


if __name__ == '__main__':
    main()
