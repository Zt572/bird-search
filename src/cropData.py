# Crops all images found in the original images directory according to their bounding box, maining the original
# structure of images.

from PIL import Image
import numpy as np
import os
import pandas as pd


# Helper method used to convert the dataset's bounding box coordinate info into the format expected from PIL for cropping
def convert_bounding_info(box_info):
    left = box_info.iloc[1]                         # X coord of top-left point
    upper = box_info.iloc[2] + box_info.iloc[4]     # Y coord of top-left point
    right = box_info.iloc[1] + box_info.iloc[3]     # X coord of bottom-right point
    lower = box_info.iloc[2]                        # Y coord of bottom-right point
    return (left, upper, right, lower)


# Import and process data
data_dir = '../data/CUB_200_2011/images'
output_dir = '../data/CUB_200_2011/cropped_images'

im = Image.open(r'../data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg')

# Import bounding box data from text file
boxing_data = pd.read_csv("../data/CUB_200_2011/bounding_boxes.txt", sep=" ", index_col=False)
print(boxing_data)

"""NOTE TO SELF: INDEXES OF IMAGES ARE WRONG. CROSS REFERENCE images.txt with the image path FOR THE IDs, then use the ID
TO GRAB THE RESPECTIVE BOUNDING BOX INFO"""

g_inx = 0   # Global index of images
for bird_species_directory in [f.path for f in os.scandir(data_dir) if f.is_dir()]:  # Going through each subdirectory of images...
    for img_path in os.listdir(bird_species_directory):  # For each image path
        with Image.open(f"{bird_species_directory}/{img_path}") as image:
            print(image.size)
            converted_coords = convert_bounding_info(boxing_data.iloc[g_inx,:])
            print(converted_coords)
            exit()
            cropped_img = image.crop(converted_coords)
            cropped_img.show()
            exit()
    exit()
