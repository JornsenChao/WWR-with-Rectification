import os
print("当前工作目录:", os.getcwd())
# sys.path.append(os.getcwd())

from src.rectification import *

img_folder = 'data_for_demo/image'
mask_folder = 'data_for_demo/mask'

temp_img_list = get_filenames_without_type(temp_img_folder)
temp_img_set = get_unique_image_id(temp_img_folder)
temp_mask_list = get_filenames_without_type(temp_mask_folder)
temp_mask_set = get_unique_image_id(temp_mask_folder)

rectify_wwr_pipeline('53533404')