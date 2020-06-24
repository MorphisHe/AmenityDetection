'''
This file parses targeted CoCo annotation file and return a Categories obj
then compare them with selected labels from open image.

Outputs set of label in CoCo that is similar to open image label in terminal.

Run Guide:
    run with CLI
    -f "file_name" : file name to process, (not path)
    -p : print flag

typical run command: python3 similar_labels.py -f instances_val2017.json -p
'''

import sys, getopt
import json
from difflib import SequenceMatcher


class Categories:
    def __init__(self):
        '''
        3 levels dict:
            level1: superclass: dict
            level2: class: dict
            level3: id: list
        '''
        self.cat_dict = {}
        self.label_list = []
    # check if superclass exist in dict
    def check_superclass(self, superclass):
        return superclass in self.cat_dict.keys()
    # check is class exist in a superclass
    def check_class(self, superclass, class_name):
        return class_name in self.cat_dict[superclass].keys()
    # check if id in is a class
    def check_id(self, superclass, class_name, class_id):
        return class_id in self.cat_dict[superclass][class_name]
    
    def add_superclass(self, superclass):
        if not self.check_superclass(superclass):
            self.cat_dict[superclass] = {}
    def add_class(self, superclass, class_name):
        if not self.check_superclass(superclass):
            self.add_superclass(superclass)
        if not self.check_class(superclass, class_name):
            self.cat_dict[superclass][class_name] = []
        self.label_list.append(class_name)
    def add_id(self, superclass, class_name, class_id):
        if not self.check_superclass(superclass):
            self.add_superclass(superclass)
        if not self.check_class(superclass, class_name):
            self.add_class(superclass, class_name)
        if not self.check_id(superclass, class_name, class_id):
            self.cat_dict[superclass][class_name].append(class_id)

    def __str__(self):
        total_superclass = 0
        # iter super class
        for superclass in self.cat_dict.keys():
            total_superclass += 1
            total_class = 0
            print("Superclass:", superclass)
            for class_name in self.cat_dict[superclass].keys():
                print("  -Class:", class_name)
                total_class += 1
                for class_ids in self.cat_dict[superclass][class_name]:
                    #for _id in class_ids:
                    print("    -id:", class_ids)
            print("  -Total Class for " + class_name + ":", total_class, "\n")
        print("Total Superclass:", total_superclass)
        return '-----------------------Done-----------------------'


def get_categories(filename, superclass_list, print_flag):
    c = None
    if '.json' in filename:
        filename = "annotations/"+filename
        fd = open(filename)
        data = json.load(fd)
        c = Categories()

        if print_flag:
            print("\nprocessing:", filename)
            
        for cat_set in data["categories"]:
            if cat_set["supercategory"] in superclass_list:
                c.add_superclass(cat_set["supercategory"])
                c.add_class(cat_set["supercategory"], cat_set["name"])
                c.add_id(cat_set["supercategory"], cat_set["name"], cat_set["id"])

    if print_flag:
        print("\n\n=================================================")
        print(c)
    
    return c


'''
compare label from coco dataset with Amenity labels I choose from 
Open Image and output ones that are similar.
'''

# outputs similar between string a and b in range [0, 1]
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


# normalize str into all lowercase with no special chars
def normalize(str):
    return str.lower().replace("_", " ")


# normalize all labels in label list
def normalize_all(label_list):
    for i in range(len(label_list)):
        label_list[i] = normalize(label_list[i])
    return label_list


OI_labels = ["Toilet",
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
        "Wine_rack"]

CoCo_labels = []


print_flag = 0
file_name = ""
opts, args = getopt.getopt(sys.argv[1:], "f:p")
for opt, arg in opts:
    if opt == "-f":
        file_name = arg
    elif opt == "-p":
        print_flag = 1





# picked superclasses
superclass_list = ["kitchen", "furniture", "electronic", "appliance", "indoor"]

# get categories obj with supercategory in superclass_list
c = get_categories(file_name, superclass_list, print_flag)
CoCo_labels = c.label_list





# normalize all labels Open Image and CoCo
OI_labels = normalize_all(OI_labels)
CoCo_labels = normalize_all(CoCo_labels)


# iterate through each of CoCo labels and get ones with >= 50% similarity
output_labels = []
for label in CoCo_labels:
    ratios = [similar(label, subset_label) for subset_label in OI_labels]
    max_ratio = max(ratios)

    # if max_ratio >= 0.55:
    output_labels.append(label)
    OI_correspondence = OI_labels[ratios.index(max_ratio)]
    
    print("Found Label: {}\nOpen Image Correspondence: {}\nSimilarity: {}%".format(label, OI_correspondence, round(max_ratio*100, 2)))
    print("==============================================")

print("\nAll Labels:\n", output_labels)


'''
Discard:
    ['laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'book',
    'teddy bear']
    these are unrelated labels, so I discard them.

Output: 
    ['bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 
    'tv', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
    'clock', 'vase', 'scissors',  'hair drier', 'toothbrush']
'''