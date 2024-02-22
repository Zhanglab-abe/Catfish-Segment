import os
import csv
import numpy as np
from skimage.io import imread

# Define the directory path
# directory = '/home/agricoptics/Desktop/CatFish/mmsegmentation/data/ade/ADEChallengeData2016/annotations/training/'
# directory = '/home/agricoptics/Desktop/CatFish/mmsegmentation/data/ade/ADEChallengeData2016/annotations/validation/'
directory = '/home/agricoptics/Desktop/CatFish/mmsegmentation/data/ade/ADEChallengeData2016/test/data_raw/annotations/'


# Define the category values
Bg_value = 1
Head_value = 2
Body_value = 3
Fins_value = 4
Tail_value = 5

# Create the CSV file
# with open('pixel_percentages_training.csv', mode='w') as csv_file:
# with open('pixel_percentages_validation.csv', mode='w') as csv_file:
with open('pixel_percentages_test.csv', mode='w') as csv_file:
    fieldnames = ['Image', 'Background', 'Head', 'Body','Fins','Tails']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Iterate through all image files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            # Load the image
            image = imread(os.path.join(directory, filename))

            # Create binary masks for each category
            Bg_mask = np.where(image == Bg_value, 1, 0)
            Head_mask = np.where(image == Head_value, 1, 0)
            Body_mask = np.where(image == Body_value, 1, 0)
            Fins_mask = np.where(image == Fins_value, 1, 0)
            Tails_mask = np.where(image == Tail_value, 1, 0)
            

            # Calculate the total number of pixels
            total_pixels = image.shape[0] * image.shape[1]

            # Calculate the number of pixels for each category
            Bg_pixels = np.sum(Bg_mask)
            Head_pixels = np.sum(Head_mask)
            Body_pixels = np.sum(Body_mask)
            Fins_pixels = np.sum(Fins_mask)
            Tail_pixels = np.sum(Tails_mask)

            # Calculate the pixel percentage for each category
            Bg_percent = (Bg_pixels / total_pixels) * 100
            Head_percent = (Head_pixels / total_pixels) * 100
            Body_percent = (Body_pixels / total_pixels) * 100
            Fins_percent = (Fins_pixels / total_pixels) * 100
            Tail_percent = (Tail_pixels / total_pixels) * 100

            # Write the pixel percentages to the CSV file
            writer.writerow({'Image': filename, 'Background': Bg_percent,
                             'Head': Head_percent, 'Body': Body_percent, 'Fins': Fins_percent, 'Tails': Tail_percent})
