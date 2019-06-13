#!/usr/bin/env bash

# Download vqa v2 dataset
VQA_DIR=./datasets/vqa
mkdir -p $VQA_DIR
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip -O $VQA_DIR/v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip -O $VQA_DIR/v2_Questions_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip -O $VQA_DIR/v2_Questions_Test_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip -O $VQA_DIR/v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip -O $VQA_DIR/v2_Annotations_Val_mscoco.zip

unzip $VQA_DIR/v2_Questions_Train_mscoco.zip -d $VQA_DIR/
unzip $VQA_DIR/v2_Questions_Val_mscoco.zip -d $VQA_DIR/
unzip $VQA_DIR/v2_Questions_Test_mscoco.zip -d $VQA_DIR/
unzip $VQA_DIR/v2_Annotations_Train_mscoco.zip -d $VQA_DIR/
unzip $VQA_DIR/v2_Annotations_Val_mscoco.zip -d $VQA_DIR/


# Unzip the BUTD features
FEAT_DIR=./datasets/coco_extract
cd $FEAT_DIR
echo Unzip train2014.tar.gz ...
tar -xzvf train2014.tar.gz
echo Unzip val2014.tar.gz ...
tar -xzvf val2014.tar.gz
echo Unzip test2015.tar.gz ...
tar -xzvf test2015.tar.gz
cd ../..


