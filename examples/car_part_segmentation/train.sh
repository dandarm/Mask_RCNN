#/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

data_path=$DIR/dataset
images_path=$data_path
annotations_path=$data_path/cogito_output_merged.json

python car_part.py\
 --images_path $images_path\
 --annotations $annotations_path\
 --checkpoint $DIR/logs/ --weights imagenet\
 --trainpercent 0.7\
 --valpercent 0.2
