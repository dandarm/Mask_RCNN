import os
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

import random
import numpy as np
from pathlib import Path
import scipy.io as sio
import tensorflow as tf
import maskrcnn.model as modellib
from maskrcnn import utils
from maskrcnn.config import Config
import imgaug.augmenters as iaa
from tqdm import tqdm
import json
from PIL import Image, ImageDraw

from pycocotools.coco import COCO
from collections import defaultdict
import random
import collections

from balancedataset import BalanceDataset
from cocoformattest import AllowedCocoData


np.random.seed(42)
#random.seet = 42
        






def prepare_datasets(part_annotation_path, images_path, train_p=0.7, val_p=0.1, test_p=0.2):

    ds = COCO(part_annotation_path)

    # dictionary to obtain annotations contained in every image
    imgToAnns = ds.imgToAnns
    # all images Id
    all_images_ids = ds.getImgIds()
    # dictionary to obtain images containing such segment category
    catToImgs = ds.catToImgs

    print("Balancing the categories among train, validation, test sets...")
    balancer = BalanceDataset(imgToAnns, all_images_ids, catToImgs)
    balancer.set_percentages(train_p, val_p, test_p)

    img_id_train, img_id_val, img_id_test = balancer.split_balanced()
    balancer.verify_split()

    dataset_train = CarPartDataset()
    id_to_category_train = dataset_train.load_dataset(part_annotation_path, images_path, img_id_train)
    dataset_train.prepare()
    print(sorted(id_to_category_train.values()))
    num_categories_train = len(id_to_category_train.items())
    print(num_categories_train)

    dataset_val = CarPartDataset()
    id_to_category_val = dataset_val.load_dataset(part_annotation_path, images_path, img_id_val)
    dataset_val.prepare()
    #print(sorted(id_to_category_val.keys()))
    num_categories_val = len(id_to_category_val.items())

    dataset_test = CarPartDataset()
    id_to_category_test = dataset_test.load_dataset(part_annotation_path, images_path, img_id_test)
    dataset_test.prepare()
    #print(sorted(id_to_category_test.keys()))
    num_categories_test = len(id_to_category_test.items())

    assert num_categories_train == num_categories_val == num_categories_test

    return dataset_train, dataset_val, dataset_test, num_categories_train



class CarPartConfig(Config):
    NAME = 'car_parts'

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 30 + 1  # 30 parts + 1 background

    # STEPS_PER_EPOCH = 100
    # VALIDATION_STEPS = 10

    # BACKBONE = "resnet50"

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 30

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    def __init__(self, num_classes):
        self.NUM_CLASSES = num_classes
        super().__init__()


class CarPartDataset(utils.Dataset):

    def load_dataset(self, annotation_json, images_dir, set_img_id):
        """
            Load the coco-like dataset from json
        Args:
            annotation_json: coco annotations file path
            images_dir: images directory
        """
        json_file = open(annotation_json)
        coco_data = json.load(json_file)
        json_file.close()

        source_name = "car_parts"

        # add class names    
        #damages_list = ['scratch', 'dent', 'severe-dent', 'substitution', 'severe_dent']
        id_to_category = {}
        for category in coco_data['categories']:
            #print(category)
            class_id = category['id']
            class_name = category['name']
            #if (class_name not in damages_list):
                ################### add_class base method from utils.Dataset
            self.add_class(source_name, class_id, class_name)
            id_to_category[class_id] = class_name

        # Get all annotations
        annotations = {}
        id_img_id_categ = {}
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            if image_id in set_img_id:
                if image_id not in annotations:
                    annotations[image_id] = []
                annotations[image_id].append(annotation)

        # Get all images 
        seen_images = {}
        num_img_not_it_annotations = 0
        id_img_not_annotated = []
        img_loaded = 0
        for image in coco_data['images']:
            image_id = image['id']
            if image_id in set_img_id:
                if image_id in seen_images:
                    print("Warning: Skipping duplicate image id: {}".format(image))
                else:
                    seen_images[image_id] = image
                    try:
                        image_file_name = image['file_name']
                        image_width = image['width']
                        image_height = image['height']
                    except KeyError as key:
                        print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))

                    image_path =  os.path.abspath(os.path.join(images_dir, image_file_name))
                    try:
                        image_annotations = annotations[image_id]

                        #### Add image base method from utils.Dataset
                        self.add_image(
                            source=source_name,
                            image_id=image_id,
                            path=image_path,
                            width=image_width,
                            height=image_height,
                            annotations=image_annotations
                        )
                    except KeyError:
                        num_img_not_it_annotations += 1
                        id_img_not_annotated.append(image_id)

                img_loaded+=1
        print (f'num_img_not_it_annotations {num_img_not_it_annotations}')
        print(id_img_not_annotated)
        print(f'Images loaded: {img_loaded}')
        return id_to_category

    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        
        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        print(mask.shape)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        return mask, class_ids


if __name__ == '__main__':
    #from keras import backend as K
    #import tensorflow.keras as keras
    ### versione di TF non GPU
    #print(K.tensorflow_backend._get_available_gpus())

    import argparse

    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect car parts')
    parser.add_argument('--images_path', required=True,
                        metavar="/path/to/car/images/",
                        help='The directory to load the images')

    parser.add_argument('--annotations_path', required=True,
                        metavar="/path/to/car/annotations/",
                        help='The directory to load the annotations')

    parser.add_argument('--weights', required=False,
                        help='the weights that can be used, values: imagenet or last')

    parser.add_argument('--checkpoint', required=True,
                        help='the folder where the checkpoints are saved')

    parser.add_argument('--epochs', required=False,
                        help='number of epochs to train')

    parser.add_argument('--lr', required=False,
                        help='the learning rate of training')

    parser.add_argument('--trainpercent', required=False,
                        help='the percentage of training set')
    parser.add_argument('--valpercent', required=False,
                        help='the percentage of validation set')
                        
    # parser.

    args = parser.parse_args()

    model_checkpoints = args.checkpoint
    print('checkpointing models in folder {}'.format(model_checkpoints))

    print('load the dataset ...')
    images_path = Path(args.images_path)
    annotations_path = Path(args.annotations_path)

    if(args.trainpercent):
        tr_percent = float(args.trainpercent)
        if(args.valpercent):
            val_percent = float(args.valpercent)
            test_percent = 1 - (tr_percent + val_percent)
            results = prepare_datasets(annotations_path, images_path, tr_percent, val_percent, test_percent)
        else:
            val_percent = 1 - tr_percent
            results = prepare_datasets(annotations_path, images_path, tr_percent, val_percent)
    else:
        results = prepare_datasets(annotations_path, images_path)

    dataset_train, dataset_val, dataset_test, num_categories = results
    print('finished loading the dataset')
    
    config = CarPartConfig(num_classes=num_categories+1)  #  + 1 because of background class

    augmentation = iaa.Sequential([
        iaa.GaussianBlur(sigma=(0.0, 5.0)),
        iaa.Affine(scale=(1., 2.5), rotate=(-90, 90), shear=(-16, 16), 
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
        iaa.LinearContrast((0.5, 1.5)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
        iaa.Alpha((0.0, 1.0), iaa.Grayscale(1.0)),
        iaa.LogContrast(gain=(0.6, 1.4)),
        iaa.PerspectiveTransform(scale=(0.01, 0.15)),
        iaa.Clouds(),
        iaa.Noop(),
        iaa.Alpha(
            (0.0, 1.0),
            first=iaa.Add(100),
            second=iaa.Multiply(0.2)),
        iaa.MotionBlur(k=5),
        iaa.MultiplyHueAndSaturation((0.5, 1.0), per_channel=True),
        iaa.AddToSaturation((-50, 50)),
    ])

    # with tf.device('/gpu:0'):
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                                model_dir=model_checkpoints)

    if args.weights == 'imagenet':
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    else:
        model.load_weights(model.find_last(), by_name=True)



    if(args.epochs):
        epoche = args.epochs
    else:
        epoche = 1
    
    if(args.lr):
        learningrate = args.lr
    else:
        learningrate = config.LEARNING_RATE
    
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=learningrate, #config.LEARNING_RATE,
                epochs=epoche,
                layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=learningrate,
                epochs=120,
                layers='4+',
                augmentation=augmentation)

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=learningrate / 10,
                epochs=160,
                layers='all',
                augmentation=augmentation)
