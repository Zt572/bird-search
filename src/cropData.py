# Crops all images found in the original images directory according to their bounding box, maintaining the original
# structure of images.

from PIL import Image
import os
import pandas as pd


# Helper method used to convert the dataset's bounding box coordinate info into the format expected from PIL for cropping
def convert_bounding_info(box_info):
    x = box_info.iloc[:,1]                              # X coord 
    y = box_info.iloc[:,2]                              # Y coord                              
    x_width = box_info.iloc[:,2] + box_info.iloc[:,4]   # X + width
    y_height = box_info.iloc[:,1] + box_info.iloc[:,3]  # Y + height
    return (float(x), float(y), float(y_height), float(x_width))


# Import and process data
data_dir = '../data/CUB_200_2011/images'
output_dir = '../data/CUB_200_2011/cropped_images'

# Import bounding box data from text file
df_box = pd.read_csv("../data/CUB_200_2011/bounding_boxes.txt", sep=" ", index_col=False)

# Get pairs of image ID to image name
df_names_ids = pd.read_csv("../data/CUB_200_2011/images.txt", sep=" ", index_col=False)

# Merge dataframes together based on shared 'id'
df_name_box = pd.merge(df_box, df_names_ids, how='inner', on='image_id')

for bird_species_directory in [f.path for f in os.scandir(data_dir) if f.is_dir()]:  # Going through each subdirectory of images...
    print("Cropping images within " + bird_species_directory.split('\\')[1] + " subdirectory...")

    for img_name in os.listdir(bird_species_directory):     # For each image 
        img_path = f"{bird_species_directory}/{img_name}"   # Create path
        name_to_match = img_path.split("\\")[1]             # Format this for direct comparison with df_names_ids table
        out_path = f"{output_dir}/{name_to_match}"          # Form output path for the given image

        if os.path.exists(out_path):    # If the cropped image already exists, then move onto the next image
            continue

        with Image.open(img_path) as image:
            # Obtain row from df_name_ids with the matching image name; this has the box info relevant to the current image
            img_id_row = df_name_box.loc[df_names_ids['image_name']==name_to_match]
            # Using box info, calculate the coordinates for cropping (conversion is necessary since PIL expects a different convention)
            converted_coords = convert_bounding_info(img_id_row)
            cropped_img = image.crop(converted_coords)
            #cropped_img.show()

            try:
                cropped_img.save(out_path)
            except FileNotFoundError:   # If the directory within cropped_images does not exist for the given species, create it
                dir_name = name_to_match.split("/")[0]  # Obtain directory name
                os.mkdir(f'../data/CUB_200_2011/cropped_images/{dir_name}')
                cropped_img.save(out_path)              # Now save it to the new directory

print("Finished! All images found within data/images have been cropped and saved to cropped_images.")