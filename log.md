### 6/9/2020: getting data from Open Image
- created new virtual env
- install awscli with pip
- pick classes:
    - subset = ["Toilet",
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

### 6/10/2020: writing a py script to query images that I wanted
- download csv files from Open Image V6:
    - class-description
    - test/valid/train annotations-bbox
- created queryImages.py
    - run with 'python3 queryOpenImagesV6.py'
    - will download 30 classes images and split into train, test, and validation sets
    - bug in queryImages.py, will modify tmr
    
### 6/11/2020: finishing getting data from open image v6
- modified downlaodOI.py to fix bug
- modified queryOpenImagesV6.py to fix bug
- all data download from open image:
    - Train: 328 mins
    - Test: 20 mins
    - Validation: 7 mins
    

