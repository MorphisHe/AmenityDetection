import detectron2
from detectron2.utils.logger import setup_logger
setup_logger() # this logs Detectron2 information such as what the model is doing when it's training

# import some common libraries
import numpy as np
import cv2
import os
import json

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor # a default predictor class to make predictions on an image using a trained model
from detectron2.config import get_cfg # a config of "cfg" in Detectron2 is a series of instructions for building a model
from detectron2.utils.visualizer import Visualizer # a class to help visualize Detectron2 predictions on an image
from detectron2.data import MetadataCatalog # stores information about the model such as what the training/test data is, what the class names are
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# Target classes with spaces removed
target_classes = ['Bathtub',
                  'Bed',
                  'Billiard table',
                  'Ceiling fan',
                  'Coffeemaker',
                  'Couch',
                  'Countertop',
                  'Dishwasher',
                  'Fireplace',
                  'Fountain',
                  'Gas stove',
                  'Jacuzzi',
                  'Kitchen & dining room table',
                  'Microwave oven',
                  'Mirror',
                  'Oven',
                  'Pillow',
                  'Porch',
                  'Refrigerator',
                  'Shower',
                  'Sink',
                  'Sofa bed',
                  'Stairs',
                  'Swimming pool',
                  'Television',
                  'Toilet',
                  'Towel',
                  'Tree house',
                  'Washing machine',
                  'Wine rack']
idx2class = {i:cls for i, cls in enumerate(target_classes)}



def _get_predictor(config_path, model_path):
    cfg = get_cfg()
    logger.info("got config")
    cfg.merge_from_file(config_path) # get baseline parameters from YAML config
    logger.info("merged")
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE='cpu'
    
    try:
        pred = DefaultPredictor(cfg)
        logger.info("created pred")
        pred.model.eval()
        logger.info("eval mode")
    except Exception as e:
        logger.error("model loading failed")
        logger.error(e) 

    return pred


def model_fn(model_dir):
    """
    Deserialize and load D2 model. This method is called automatically by Sagemaker.
    model_dir is location where your trained model will be downloaded.
    """
    try:
        # Restoring trained model, take a first .yaml and .pth/.pkl file in the model directory
        for file in os.listdir(model_dir):
            # looks up for yaml file with model config
            if file.endswith(".yaml"):
                config_path = os.path.join(model_dir, file)
            # looks up for *.pkl or *.pth files with model weights
            if file.endswith(".pth") or file.endswith(".pkl"):
                model_path = os.path.join(model_dir, file)
        
        logger.info(f"Using config file {config_path}")
        logger.info(f"Using model weights from {model_path}") 
        
        pred = _get_predictor(config_path,model_path)
        return pred
    except Exception as e:
        logger.error("Model deserialization failed...")
        logger.error(e)  
        
    logger.info("Deserialization completed ...")
    
    return None


def input_fn(request_body, request_content_type):
    """
    Converts image from NPY format to numpy.
    """
    logger.info(f"Handling inputs...Content type is {request_content_type}")
    
    try:
        if "jpeg" in request_content_type:
            nparr = np.frombuffer(request_body, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            input_object = np.asarray(img)
            return input_object
        else:
            raise Exception(f"Unsupported request content type {request_content_type}")
    except Exception as e:
        logger.error("Input deserialization failed...")
        logger.error(e)  
        return None
            
    logger.info("Input deserialization completed...")
    logger.info(f"Input object type is {type(input_object)} and shape {input_object.shape}")

    return None


def predict_fn(input_object, model):
    # according to D2 rquirements: https://detectron2.readthedocs.io/tutorials/models.html
    
    logger.info("Doing predictions...")
    logger.debug(f"Input object type is {type(input_object)} and shape {input_object.shape}")
    logger.debug(f"Predictor type is {type(model)}")
    
    try:
        prediction = model(input_object)
        return prediction
    except Exception as e:
        logger.error("Prediction failed...")
        logger.error(e)
        return None

    logger.debug("Predictions are:")
    logger.debug(prediction)
    
    return None


def output_fn(prediction, response_content_type):
    
    logger.info("Processing output predictions...")
    logger.debug(f"Output object type is {type(prediction)}")
        
    try:
        pred_classes = []
        preds = prediction["instances"].to("cpu")
        logger.info("put into cpu")
        for i in range(len(preds)):
            pred_class = preds[i].pred_classes.item()
            logger.info("pred.item")
            score = preds[i].scores.item()
            logger.info("score.item")
            if score >= 0.10:
                pred_classes.append(pred_class)
        
        pred_classes = list(set(pred_classes))
        logger.info("list(set(pred))")
        pred_classes = [idx2class[i] for i in pred_classes]
        logger.info(pred_classes)
        output = json.dumps(pred_classes)
        logger.info("pred_classes")
        return output
        
    except Exception as e:
        logger.error("Output processing failed...")
        logger.error(e)
        return None
    
    logger.info("Output processing completed")

    return None