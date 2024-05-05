import os
import cv2
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
import seaborn as sns
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import DBSCAN

from src.rectification import compute_edgelets, vis_edgelets, ransac_vanishing_point, reestimate_model, vis_model, compute_homography_and_warp,remove_inliers

def get_filenames_without_type(folder_path):
    file_list = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            # filename = filename[:]
            file_list.append(os.path.splitext(filename)[0])
    return file_list
def get_unique_image_id(folder_path):
    file_list = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            filename = filename[:]
            file_list.append(int(os.path.splitext(filename)[0].split('_')[0]))
    return file_list
def get_file_name_and_path_with_prefix(folder_path, prefix):
    filename = []
    filepath = []
    for file in os.listdir(folder_path):
        if file.startswith(prefix):
            filename.append(file)
            filepath.append(os.path.join(folder_path, file))
    filename, filepath = zip(*sorted(zip(filename, filepath),
                                     key=lambda x: int(x[0].split('_')[1].split('.')[0])))

    return list(filename), list(filepath)

def get_image_name_dict_for_bldg_id(folder_path):
    image_list = get_filenames_without_type(folder_path)
    image_list_dict = {}
    for image in image_list:
        image_id = image.split('_')[0]
        if image_id not in image_list_dict:
            image_list_dict[image_id] = []
        image_list_dict[image_id].append(image)
    return image_list_dict
def get_image_path_dict_for_bldg_id(folder_path):
    image_list_dict = {}
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            image_id = filename.split('_')[0]
            if image_id not in image_list_dict:
                image_list_dict[image_id] = []
            image_list_dict[image_id].append(os.path.join(folder_path, filename))
    return image_list_dict

def show_images_from_path_list(path_list):
    num_images = len(path_list)
    if (num_images > 1):
        fig, axes = plt.subplots(1, num_images, figsize=(num_images * 4, 4))
        for i, path in enumerate(path_list):
            img = mpimg.imread(path)
            axes[i].imshow(img)
            axes[i].axis('off')
        plt.show()
    else:
        figsize=(4, 4)
        img = mpimg.imread(path_list[0])
        plt.imshow(img)
        plt.show()
        
def show_images_from_id(id, folder_path, image_dict):
    path_list = image_dict[id]
    print(path_list)
    num_images = len(path_list)
    if (num_images > 1):
        fig, axes = plt.subplots(1, num_images, figsize=(num_images * 4, 4))
        for i, path in enumerate(path_list):
            img = mpimg.imread(path)
            axes[i].imshow(img)
            axes[i].axis('off')
        plt.show()
    else:
        figsize=(4, 4)
        img = mpimg.imread(path_list[0])
        plt.imshow(img)
        plt.show()


def compute_wwr(original_image, mask_image):
    # Check if the mask is a single-channel 8-bit image
    if mask_image.ndim != 2 or mask_image.dtype not in [np.uint8, np.int8]:
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

    # Threshold the mask to get the white areas
    _, thresholded = cv2.threshold(mask_image, 254, 255, cv2.THRESH_BINARY)

    # Find contours of the white areas
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask from the contours
    mask_of_white_area = np.zeros_like(mask_image)
    cv2.drawContours(mask_of_white_area, contours, -1, (255), thickness=cv2.FILLED)


    # Apply the mask to the original image to extract the area
    extracted_area = cv2.bitwise_and(original_image, original_image, mask=mask_of_white_area)

    # Calculate the number of non-zero pixels in the white area mask
    total_white_area_pixels = cv2.countNonZero(mask_of_white_area)

    # Create a mask for the non-white area within the white contours (windows)
    mask_of_windows = cv2.bitwise_xor(mask_of_white_area, thresholded)

    # Calculate the number of non-zero pixels in the windows mask
    total_window_pixels = cv2.countNonZero(mask_of_windows)
    WWR = (total_window_pixels/total_white_area_pixels)
    # Save and display the resulting image
    # extracted_image_path = '/mnt/data/extracted_area.jpg'
    # cv2.imwrite(extracted_image_path, extracted_area)

    # extracted_image_path, 
    # print(total_white_area_pixels, total_window_pixels, f"{WWR:.2%}")

    plt.imshow(extracted_area)
    plt.imshow(mask_of_windows)
    plt.show()
    return f"{WWR:.2%}"

def rectify_image_and_mask_based_on_image(image, mask):
    # 计算edgelets并可视化
    edgelets1 = compute_edgelets(image)
    # vis_edgelets(image, edgelets1)  # Visualize the edgelets

    # 计算第一个消失点并重估计
    vp1 = ransac_vanishing_point(edgelets1, num_ransac_iter=50000, threshold_inlier=5)
    vp1 = reestimate_model(vp1, edgelets1, threshold_reestimate=1)
    vis_model(image, vp1)  # Visualize the vanishing point model

    # 移除内点，计算第二个消失点
    edgelets2 = remove_inliers(vp1, edgelets1, 10)
    vp2 = ransac_vanishing_point(edgelets2, num_ransac_iter=2000, threshold_inlier=5)
    vp2 = reestimate_model(vp2, edgelets2, threshold_reestimate=1)
    vis_model(image, vp2)  # Visualize the vanishing point model

    # 应用单应性变换并可视化结果
    clip_factor = 4  # 调整此参数以适应图像尺寸
    warped_img = compute_homography_and_warp(image, vp1, vp2, clip_factor=clip_factor)
    warped_mask = compute_homography_and_warp(mask, vp1, vp2, clip_factor=clip_factor)
    return warped_img, warped_mask

def rectify_image_and_mask_based_on_mask(image, mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 计算edgelets并可视化
    edgelets1 = compute_edgelets(mask)
    # vis_edgelets(image, edgelets1)  # Visualize the edgelets

    # 计算第一个消失点并重估计
    vp1 = ransac_vanishing_point(edgelets1, num_ransac_iter=50000, threshold_inlier=5)
    vp1 = reestimate_model(vp1, edgelets1, threshold_reestimate=1)
    vis_model(mask, vp1)  # Visualize the vanishing point model

    # 移除内点，计算第二个消失点
    edgelets2 = remove_inliers(vp1, edgelets1, 10)
    vp2 = ransac_vanishing_point(edgelets2, num_ransac_iter=2000, threshold_inlier=5)
    vp2 = reestimate_model(vp2, edgelets2, threshold_reestimate=1)
    vis_model(mask, vp2)  # Visualize the vanishing point model

    # 应用单应性变换并可视化结果
    clip_factor = 4  # 调整此参数以适应图像尺寸
    warped_img = compute_homography_and_warp(image, vp1, vp2, clip_factor=clip_factor)
    warped_mask = compute_homography_and_warp(mask, vp1, vp2, clip_factor=clip_factor)
    return warped_img, warped_mask

def rectify_wwr_pipeline(img_folder, mask_folder, bldg_osm_id):
    # load path and list from this osm_id building, image and mask
    img_list, img_path = get_file_name_and_path_with_prefix(img_folder, bldg_osm_id)
    mask_list, mask_path = get_file_name_and_path_with_prefix(mask_folder, bldg_osm_id)
    # create a list of wwr for various image of this building
    wwr_list = {}
    for i in range(len(img_path)):
        wwr_dict = {}
        # load image and mask
        original_image  = cv2.imread(img_path[0])
        mask_image  = cv2.imread(mask_path[0], cv2.IMREAD_GRAYSCALE)
        # compute the original wwr without rectification
        wwr = compute_wwr(original_image, mask_image)
        wwr_dict['wwr'] = wwr
        
        # rectification   
        # warped_img, warped_mask = rectify_image_and_mask_based_on_image(original_image, mask_image)
        warped_img, warped_mask = rectify_image_and_mask_based_on_mask(original_image, mask_image)
        # compute the wwr after rectification
        warped_mask = cv2.convertScaleAbs(warped_mask, alpha=255)
        warped_wwr = compute_wwr(warped_img, warped_mask)
        wwr_dict['warped_wwr'] = warped_wwr

        wwr_list [i] = wwr_dict
    return wwr_list