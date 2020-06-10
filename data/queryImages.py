# this file uses downloadOI.py to get all data with label in subset.
# data will be split into train, validation, and test

import os
import time

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
            "Kitchen_" + "\&" + "_dining_room_table",
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
subset = ["Kitchen_&_dining_room_table"]

subset_classes = ','.join(subset)
modes = ['validation'] #'validation', 'test', 

for mode in modes:
    start_time = time.time()
    print("\n\n=============== Working on " + mode + " data ===============")
    cmd_to_excute = "python3 downloadOI.py --classes '" + subset_classes + "' --mode " + mode
    print("==============Excuting:", cmd_to_excute, "==============\n")
    os.system(cmd_to_excute)
    end_time = time.time()
    print("==============Total Time:", (end_time-start_time)//60, "mins==============")