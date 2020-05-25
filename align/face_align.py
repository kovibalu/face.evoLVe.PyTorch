from PIL import Image
from detector import detect_faces, load_detect_faces_models
from align_trans import get_reference_facial_points, warp_and_crop_face, crop_face
import numpy as np
import os
from tqdm import tqdm
import argparse


def process_img(det_models, img_path, img_out_path, max_size=1024):
    print("Processing\t{}".format(img_path))
    img = Image.open(img_path)
    img.thumbnail((max_size, max_size))
    print("Size: {}".format(img.size))
    try: # Handle exception
        bounding_box, landmarks = detect_faces(det_models, img)
    except Exception as e:
        print("{} is discarded due to exception!".format(os.path.join(source_root, subfolder, image_name)))
        print("--- {}".format(e))
        img.close()
        return

    if len(landmarks) == 0: # If the landmarks cannot be detected, the img will be discarded
        #print("{} is discarded due to non-detected landmarks!".format(os.path.join(source_root, subfolder, image_name)))
        print("{} - has non-detected landmarks!".format(img_path))
        print("Filaneme is {}".format(img_out_path))
        print(bounding_box)
        """
        print('bounding box is:')
        print(bounding_box)
        cropped_face_list = crop_face(np.array(img),bounding_box, crop_size=(crop_size, crop_size))
        print(cropped_face_list)
        for cropped_face in cropped_face_list:
            print(cropped_face)
            img_cropped = Image.fromarray(cropped_face)
            if image_name.split('.')[-1].lower() not in ['jpg', 'jpeg']: #not from jpg
                image_name = '.'.join(image_name.split('.')[:-1]) + '.jpg'
            print('dest is {}'.format(img_out_path))
            img_cropped.save(img_out_path)
        """
        """
        img_cropped = Image.fromarray(cropped_face)
        if image_name.split('.')[-1].lower() not in ['jpg', 'jpeg']: #not from jpg
            image_name = '.'.join(image_name.split('.')[:-1]) + '.jpg'
        img_cropped.save(img_out_path)
        #continue
        """
    else:
        facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
        warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
        img_warped = Image.fromarray(warped_face)
        if img_out_path.split('.')[-1].lower() not in ['jpg', 'jpeg']: #not from jpg
            img_out_path = '.'.join(img_out_path.split('.')[:-1]) + '.jpg'
        img_warped.save(img_out_path)
        img_warped.close()

    img.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "face alignment")
    parser.add_argument("-source_root", "--source_root", help = "specify your source dir", default = "./data/test", type = str)
    parser.add_argument("-dest_root", "--dest_root", help = "specify your destination dir", default = "./data/test_Aligned", type = str)
    parser.add_argument("-crop_size", "--crop_size", help = "specify size of aligned faces, align and crop with padding", default = 112, type = int)
    args = parser.parse_args()

    source_root = args.source_root # specify your source dir
    dest_root = args.dest_root # specify your destination dir
    crop_size = args.crop_size # specify size of aligned faces, align and crop with padding
    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square = True) * scale

    cwd = os.getcwd() # delete '.DS_Store' existed in the source_root
    os.chdir(source_root)
    os.system("find . -name '*.DS_Store' -type f -delete")
    os.chdir(cwd)

    if not os.path.isdir(dest_root):
        os.mkdir(dest_root)

    print("Loading face detection models...")
    det_models = load_detect_faces_models()

    for subfolder in tqdm(os.listdir(source_root)):
        if not os.path.isdir(os.path.join(dest_root, subfolder)):
            os.mkdir(os.path.join(dest_root, subfolder))
        for image_name in os.listdir(os.path.join(source_root, subfolder)):
            img_path = os.path.join(source_root, subfolder, image_name)
            img_out_path = os.path.join(dest_root, subfolder, image_name)
            process_img(
                det_models=det_models,
                img_path=img_path,
                img_out_path=img_out_path)
