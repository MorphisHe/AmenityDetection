# this file uses downloadOI.py to get all data with label in subset.
# data will be split into train, validation, and test
# run with "python3 queryOpenImagesV6.py"

import os
import time
import shutil

subset = ["Toilet",
            "Swimming_pool",
            "Bed",
            "Billiard_table",
            "Sink",
            "Fountain",
            "Oven",
            "Ceiling_fan",
            "Television",
            "Microwave_oven",
            "Gas_stove",
            "Refrigerator",
            "Kitchen_&_dining_room_table",
            "Washing_machine",
            "Bathtub",
            "Stairs",
            "Fireplace",
            "Pillow",
            "Mirror",
            "Shower",
            "Couch",
            "Countertop",
            "Coffeemaker",
            "Dishwasher",
            "Sofa_bed",
            "Tree_house",
            "Towel",
            "Porch",
            "Wine_rack",
            "Jacuzzi"]

subset_classes = ','.join(subset)
modes = ['train', 'validation', 'test'] #'validation', 'test', 

for mode in modes:
    start_time = time.time()
    print("\n\n=============== Working on " + mode + " data ===============")
    cmd_to_excute = "python3 downloadOI.py --classes '" + subset_classes + "' --mode " + mode
    print("==============Excuting:", cmd_to_excute, "==============\n")
    os.system(cmd_to_excute)
    end_time = time.time()
    print("==============Total Time:", (end_time-start_time)//60, "mins==============")


# since all "Kitchen_&_dining_room_table" images is in data dir, move them to correct dir
dir_names = ['train', 'validation', 'test']

for dir_name in dir_names:
    directory = os.fsencode(dir_name)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        
        if 'jpg' in filename:
            # move image to "Kitchen_&_dining_room_table" folder
            shutil.move(dir_name+'/'+filename, dir_name+'/Kitchen_&_dining_room_table')