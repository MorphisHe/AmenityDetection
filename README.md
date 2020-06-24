# Website for detecting amenities in a picture

### Project Time: 6/8/2020 to recent
- current task: getting data

### Problem To Solve
There are many websites online such as house rental, where house owners post pictures of the property for customers to see. It is always good when owners include types of amenity as description. However, not all owner does that. So implmeneting an object detector to detect all types of amenities in each picture can save lots of time for those house owners, and thus improve the customer experience.

### Image Source: Open Image

### Sources
- article: https://medium.com/airbnb-engineering/amenity-detection-and-beyond-new-frontiers-of-computer-vision-at-airbnb-144a4441b72e

### Todos
- change Microwave_oven in Open Image to microwave
- change Kitchen_&_dining_room_table to dining table in Open Image

- get dataset (Done)
    - (Open Image V4 https://storage.googleapis.com/openimages/web/index.html)
    - contains 600 classes, filter out ones related to amenity (about 30)
    - about 32k images from 30 classes
    - Google data labeling service: helps label new data
- model
    - (Tensorflow Object Detection API)
        - tfrecords
        - tensorboard (monitor training process)
        - Tutorial: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/
    - Pre-Trained Model (exploring, pick one)
        - ssd_mobilenet_v2 (fast, lower accuracy)
        - faster_rcnn_inception_resnet_v2 (slower, higher accuracy)
        - use mean average percision (mAP) as metric. (standard for obj detection model)
            - tutorial :https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
        - fine-tunning tricks: refer to the article
        - Google AutoML Vision:
            - auto-training
            - can be better than fine-tunning

