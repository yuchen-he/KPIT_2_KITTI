from __future__ import print_function
import argparse
import os
import cv2


def parse_arguments():
    # parse the arguments
    parser = argparse.ArgumentParser(description='Do resize and crop for 8M image')
    parser.add_argument(
        "--input_path", "-in",
        default="./original_input/",
        help="path to original image folder",
    )
    parser.add_argument(
        "--save_path", "-s",
        default="./resized_output_1024×320/",
        help="path to save resized image",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    # word_dir = os.getcwd()
    img_list = os.listdir(args.input_path)
    for img_path in img_list:
        img_path_full = os.path.join(args.input_path, img_path)
        img_orig = cv2.imread(img_path_full)
        # (3840, 1920) -> (1280, 640) -> (1248, 384)
        # img_resized = cv2.resize(img_orig, (1280, 640), interpolation=cv2.INTER_CUBIC)
        # img_crop = img_resized[128:512, 16:1264, :]
        # (3840, 1920) -> (1024, 512) -> (1024, 320)
        img_resized = cv2.resize(img_orig, (1024, 512), interpolation=cv2.INTER_CUBIC)
        img_crop = img_resized[96:416, :, :]
        save_path = os.path.join(args.save_path, img_path)
        cv2.imwrite(save_path, img_crop)
        print('After resized: ', img_resized.shape)
        print('After crop: ', img_crop.shape)


if __name__ == '__main__':
    main()
