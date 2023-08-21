import os
import json
from preprocess import load_json_labels
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo # a series of pre-trained Detectron2 models: https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md



def register_dataset(image_dirs, cls_to_idx):
    # Register datasets with Detectron2 (required by Detectron2 library)
    print("\n\n")
    print("*"*20)
    print("Registering datasets:")
    for dataset_name in image_dirs:
        # Create dataset name strings
        print(f"Registering {dataset_name}")

        # Register the datasets with Detectron2's DatasetCatalog, which has space for a lambda function to preprocess it
        DatasetCatalog.register(dataset_name, lambda dataset_name=dataset_name: load_json_labels(dataset_name))

        # Create the metadata for our dataset (the main thing being the classnames we're using)
        MetadataCatalog.get(dataset_name).set(thing_classes=sorted(list(cls_to_idx.keys())))


def train(image_dirs, cls_to_idx):
    print("\n\n")
    print("*"*20)
    print("Start Training:")
    # Setup a model config (recipe for training a Detectron2 model)
    cfg=get_cfg()

    # Add some basic instructions for the Detectron2 model from the model_zoo: https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))

    # Add some pretrained model weights from an object detection model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")

    # Setup datasets to train/validate on (this will only work if the datasets are registered with DatasetCatalog)
    cfg.DATASETS.TRAIN = (image_dirs[0],) # train dir
    cfg.DATASETS.TEST = (image_dirs[1],) # valid dir

    # How many dataloaders to use? This is the number of CPUs to load the data into Detectron2, set to max
    cfg.DATALOADER.NUM_WORKERS = os.cpu_count()

    # How many images per batch
    cfg.SOLVER.IMS_PER_BATCH = 2

    # the original model used 0.01 because of batch_size=16, so we'll divide by 8: 0.01/8 = 0.00125.
    cfg.SOLVER.BASE_LR = 0.00125

    # How many iterations are we going for? (300 is okay for our small model, increase for larger datasets)
    cfg.SOLVER.MAX_ITER = 300

    # ROI = region of interest, as in, how many parts of an image are interesting, how many of these are we going to find? 
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    # number of classes
    cfg.MODEL.RETINANET.NUM_CLASSES = len(cls_to_idx)

    # Setup output directory, all the model artefacts will get stored here in a folder called "outputs" 
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Setup the default Detectron2 trainer, see: https://detectron2.readthedocs.io/modules/engine.html#detectron2.engine.defaults.DefaultTrainer
    trainer = DefaultTrainer(cfg)

    # Resume training from model checkpoint or not, we're going to just load the model in the config: https://detectron2.readthedocs.io/modules/engine.html#detectron2.engine.defaults.DefaultTrainer.resume_or_load
    trainer.resume_or_load(resume=False) 

    # Start training
    trainer.train()


if __name__ == "__main__":
    CLS_TO_IDX_PATH = "data_collection/cls_to_idx.json" # path to the class name to index mapping json file
    IMAGE_DIRS = ["data_collection/train/", "data_collection/validation/"]
    
    # load class to index mapping
    cls_to_idx = json.load(open(CLS_TO_IDX_PATH))

    # register dataset
    register_dataset(IMAGE_DIRS, cls_to_idx)

    # train model
    train(IMAGE_DIRS, cls_to_idx)