import numpy as np
from PIL import Image
import os
import sys

from align.detector import detect_faces, load_detect_faces_models
from align.visualization_utils import show_results
from util.extract_feature_v2 import extract_feature_for_img, load_face_id_model

IMG_PATH = '/home/ec2-user/images/neelam_photos/IMG_0143.jpg'
SAVE_FILE_PATH = '/home/ec2-user/projects/facelab-data/test_Aligned/'


def main(img_path):
    img = Image.open(img_path) # modify the image path to yours
    det_models = load_detect_faces_models()
    bounding_boxes, landmarks = detect_faces(det_models, img) # detect bboxes and landmarks for all faces in the image
    
    viz_img = show_results(img, bounding_boxes, landmarks) # visualize the results
    viz_img.save('out.jpg')

    face_id_model = load_face_id_model()

    for bbox_idx, bbox in enumerate(bounding_boxes):
        tl_x, tl_y, br_x, br_y, prob = bbox.astype(int)
        cropped_img = np.array(img)[tl_y:br_y, tl_x:br_x, :]
        Image.fromarray(cropped_img).save('bbox-{}.jpg'.format(bbox_idx))
        features = extract_feature_for_img(
            img=cropped_img,
            backbone=face_id_model)
        print(features)


def create_train_features(img_dir,dest_dir,dest_filename):
    '''
    img_dir: dir where there are cropped images of kovi or neelam
    extract_feature_for_img: returns a tensor object
    convert tensor obj to numpy array for each image
    save these numpy arrays to a list
    stack this list
    save to a file that contains  an numpy array that contains vectors  for each image
    each image vector is a row in the array
    '''
    
    face_id_model = load_face_id_model()
    features_list = []
    for img_name in os.listdir(img_dir):
        img_full_name = os.path.join(img_dir,img_name)
        img = Image.open(img_full_name)
        features = extract_feature_for_img(
            img=np.array(img),
            backbone=face_id_model)
        img.close()
        #features is a tensor so convert tensor to numpy array 
        features_list.append(features.numpy())
    final_array = np.stack(features_list)
    #print(final_array.size)
    np.save(dest_filename, final_array)



if __name__ == '__main__':
    #main(
    #    img_path=IMG_PATH,
    #    face_id_model_root=FACE_ID_MODEL_ROOT)
    img_dir = '/home/ec2-user/projects/facelab-data/test_Aligned/neelam'
    dest_dir = '/home/ec2-user/projects/facelab-data/test_Aligned/'
    dest_filename = os.path.join(dest_dir,sys.argv[1])

    create_train_features(img_dir, 
    dest_dir=dest_dir,
    dest_filename=dest_filename)
