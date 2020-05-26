import argparse
import os
import numpy as np
from tqdm import tqdm

from PIL import Image

from align.align_trans import get_reference_facial_points
from align.detector import load_detect_faces_models, process_faces


def process_img(det_models,
                reference,
                crop_size,
                img_path,
                img_out_path,
                max_size=1024):
    print("Processing\t{}".format(img_path))
    img = Image.open(img_path)
    img.thumbnail((max_size, max_size))
    face_results = process_faces(
        img=img,
        det_models=det_models,
        reference=reference,
        crop_size=crop_size)
    img.close()

    for i, face_result in enumerate(face_results):
        img_warped = Image.fromarray(face_result.warped_face)
        img_without_ext = '.'.join(img_out_path.split('.')[:-1])
        img_out_new_path = "{}-{}.jpg".format(img_without_ext, i)
        img_warped.save(img_out_new_path)
        img_warped.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="face alignment")
    parser.add_argument("-source_root", "--source_root",
                        help="specify your source dir", default="./data/test", type=str)
    parser.add_argument("-dest_root", "--dest_root", help="specify your destination dir",
                        default="./data/test_Aligned", type=str)
    parser.add_argument("-crop_size", "--crop_size",
                        help="specify size of aligned faces, align and crop with padding", default=112, type=int)
    args = parser.parse_args()

    source_root = args.source_root  # specify your source dir
    dest_root = args.dest_root  # specify your destination dir
    crop_size = args.crop_size  # specify size of aligned faces, align and crop with padding
    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square=True) * scale

    cwd = os.getcwd()  # delete '.DS_Store' existed in the source_root
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
            print(img_out_path)
            process_img(
                det_models=det_models,
                reference=reference,
                crop_size=crop_size,
                img_path=img_path,
                img_out_path=img_out_path)
