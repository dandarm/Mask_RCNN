#/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

data_path=$DIR/dataset
images_path=$data_path/VOCdevkit/VOC2010/JPEGImages
annotations_path=$data_path/car_part_annotations/Annotations_Part

dataset_path_train=/tf/isa-sc-acde-utils/WIN/IN/Datasets/segmentation/data_train.json
dataset_path_val=/tf/isa-sc-acde-utils/WIN/IN/Datasets/segmentation/data_val.json
dataset_path_test=/tf/isa-sc-acde-utils/WIN/IN/Datasets/segmentation/data_test.json
img_path=/tf/isa-sc-acde-utils/WIN/IN/Datasets/retraining_032020

echo "training ${dataset_path}"

python car_part.py\
 --images_path $images_path\
 --annotations $annotations_path\
 --checkpoint $DIR/logs/ --weights imagenet\
 --trainpercent 0.1\
 --valpercent 0.1


CUDA_VISIBLE_DEVICES=1 python $DIR/car_part.py\
 --annotations_path_train $dataset_path_train\
 --annotations_path_val $dataset_path_val\
 --annotations_path_test $dataset_path_test\
 --epochs 1\
 --images_path $img_path\
 --weights "imagenet"\
 --checkpoint /tf/isa-sc-acde-utils/WIN/IN/Datasets/segmentation/checkpoint