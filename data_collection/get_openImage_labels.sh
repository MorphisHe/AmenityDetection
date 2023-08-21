# get openImage-v5 labels
# Cmd: bash get_openImage_labels.sh
# result will store in labels/ directory

# Training bounding boxes (1.11G)
wget https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv -P labels

# Validating bounding boxes (23.94M)
wget https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv -P labels
    
# Testing bounding boxes (73.89M)
wget https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv -P labels

# Class names of images (11.73K)
wget https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv -P labels