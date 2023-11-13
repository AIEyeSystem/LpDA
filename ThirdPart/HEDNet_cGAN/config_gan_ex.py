#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Author  : Qiqi Xiao & Jiaxu Zou
# @Email     : xiaoqiqi177@gmail.com & zoujx96@gmail.com
# @File    : config_gan_ex.py
# **************************************
LESION_IDS = {'EX':0, 'HE':1, 'MA':2, 'SE':3}

#Modify the general parameters.
# IMAGE_DIR = '../../../../../data/IDRiD/seg/'
IMAGE_DIR = '../../data/IDRiD/seg/'
# IMAGE_DIR = 'tmp/gradnpy/img/'
LESION_NAME = 'EX'
CLASS_ID = LESION_IDS[LESION_NAME]
NET_NAME = 'hednet'
PREPROCESS = True
IMAGE_SIZE = 256

#Modify the parameters for training.
EPOCHES = 5000
TRAIN_BATCH_SIZE = 4
D_WEIGHT = 0.01
D_MULTIPLY = False
PATCH_SIZE = 16
MODELS_DIR = 'results/models_ex'
LOG_DIR = 'drlog_hednet_true_ex_gan'
G_LEARNING_RATE = 0.001
D_LEARNING_RATE = 0.001
LESION_DICE_WEIGHT = 0.
ROTATION_ANGEL = 20
CROSSENTROPY_WEIGHTS = [0.1, 1.]
RESUME_MODEL = None
