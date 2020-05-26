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


def main():
    # cap = cv2.VideoCapture(0)
    print('Loading models...')
    det_models = load_detect_faces_models()
    face_id_model = load_face_id_model()

    crop_size = 112
    max_size = 1024
    reference = get_reference_facial_points(default_square=True)

    print('Starting image processing...')
    image_names = list(os.listdir(STREAM_DIR))
    for img_idx, image_name in tqdm(enumerate(image_names)):
        pil_img = Image.open(os.path.join(STREAM_DIR, image_name))
        if False:
            ret, image_np = cap.read()
            # BGR -> RGB
            pil_img = Image.fromarray(image_np[..., ::-1])

        pil_img.thumbnail((max_size, max_size))
        # Detect bboxes and landmarks for all faces in the image and warp the faces.
        face_results = process_faces(
            img=pil_img,
            det_models=det_models,
            reference=reference,
            crop_size=crop_size)

        for fr_idx, face_result in enumerate(face_results):
            features = extract_feature_for_img(
                img=face_result.warped_face,
                backbone=face_id_model)

        names = ['Neelam'] * len(face_results)

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
            names=names)
        viz_img.save('{}-stream.jpg'.format(img_idx))


if __name__ == '__main__':
    main()
