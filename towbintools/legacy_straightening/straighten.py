import sys

sys.path.append(".")
import os

from joblib import Parallel, delayed
from straightening_tools import straighten_video_from_backbone
from tifffile import imread, imwrite


def run_and_save(source_video, backbone_video, output_dir, width):
    print(f'### Straightening {os.path.basename(source_video)} ###')
    source_vid = imread(source_video)
    backbone_vid = imread(backbone_video)
    straightened_video = straighten_video_from_backbone(source_vid, backbone_vid, width)
    imwrite(os.path.join(output_dir, os.path.basename(source_video)), straightened_video, compression = 'zlib')

def get_images_of_point(images, point):
    """
    Given a list of images and a point, return a list of image names that contain the point in their file names.
    
    Parameters:
        images (list): List of image file names.
        point (str): Point to search for in image file names.
    
    Returns:
        list: List of image names that contain the point in their file names.
    """
    
    # Initialize empty list to store image names
    image_names = []
    
    # Iterate through list of images
    for image in images:
        # Check if point is in the image file name
        if point in os.path.basename(image):
            # If point is found, append image name to list
            image_names.append(image)
    
    # Return list of image names
    return image_names

if __name__ == '__main__':
    pharynx_skel_videos_dir = r"/mnt/external.data/TowbinLab/kstojanovski/20220629_Ti2_20x_160-182-190_pumping_25C_20220629_154238_325/analysis/test_pharynx_skel_videos/"
    pharynx_videos_dir = r"/mnt/external.data/TowbinLab/kstojanovski/20220629_Ti2_20x_160-182-190_pumping_25C_20220629_154238_325/analysis/cropped_pharynx_videos/"
    output_dir = r"/mnt/external.data/TowbinLab/kstojanovski/20220629_Ti2_20x_160-182-190_pumping_25C_20220629_154238_325/analysis/test_str_pharynx_videos/"

    os.makedirs(output_dir, exist_ok = True)

    pharynx_skel_videos = sorted([os.path.join(pharynx_skel_videos_dir, video) for video in os.listdir(pharynx_skel_videos_dir)])
    pharynx_videos = sorted([os.path.join(pharynx_videos_dir, video) for video in os.listdir(pharynx_videos_dir)])

    pharynx_videos = sorted(get_images_of_point(pharynx_videos, 'Point0006'))
    Parallel(n_jobs = 10)(delayed(run_and_save)(pharynx_video, pharynx_skel_video, output_dir, 141) for pharynx_skel_video, pharynx_video in zip(pharynx_skel_videos, pharynx_videos))