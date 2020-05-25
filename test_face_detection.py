import numpy as np
from PIL import Image

from align.detector import detect_faces, load_detect_faces_models
from align.visualization_utils import show_results
from util.extract_feature_v2 import extract_feature_for_img, load_face_id_model
from util.model_irse import IR_50

IMG_PATH = '/home/ec2-user/images/neelam_photos/IMG_0143.jpg'
FACE_ID_MODEL_ROOT = '/home/ec2-user/projects/facelab-data/models/backbone_ir50_ms1m_epoch120.pth'


def main(img_path, face_id_model_root):
    img = Image.open(img_path) # modify the image path to yours
    det_models = load_detect_faces_models()
    bounding_boxes, landmarks = detect_faces(det_models, img) # detect bboxes and landmarks for all faces in the image
    print(bounding_boxes)
    print(type(bounding_boxes))
    print(bounding_boxes[0][4])
    viz_img = show_results(img, bounding_boxes, landmarks) # visualize the results
    viz_img.save('out.jpg')

    face_id_model = load_face_id_model(
        backbone=IR_50([112, 112]),
        model_root=FACE_ID_MODEL_ROOT)

    for bbox_idx, bbox in enumerate(bounding_boxes):
        tl_x, tl_y, br_x, br_y, prob = bbox.astype(int)
        cropped_img = np.array(img)[tl_y:br_y, tl_x:br_x, :]
        Image.fromarray(cropped_img).save('bbox-{}.jpg'.format(bbox_idx))
        features = extract_feature_for_img(
            img=cropped_img,
            backbone=face_id_model)
        print(features)


if __name__ == '__main__':
    main(
        img_path=IMG_PATH,
        face_id_model_root=FACE_ID_MODEL_ROOT)
