"""
This scripts demos how to do single video classification using the framework
Before using this scripts, please download the model files using

bash models/get_reference_models.sh

Usage:

python classify_video.py <video name>
"""

import os
anet_home = os.environ['ANET_HOME']
import sys
sys.path.append(anet_home)

from pyActionRec.action_classifier import ActionClassifier
from pyActionRec.anet_db import ANetDB
import argparse
import numpy as np
import json

parser = argparse.ArgumentParser()
parser.add_argument("video_name", type=str)
parser.add_argument("--video_list", action="store_true", default=False)
parser.add_argument("--use_flow", action="store_true", default=False)
parser.add_argument("--save_json", action="store_true", default=False)
parser.add_argument("--gpu", type=int, default=0)

args = parser.parse_args()

VIDEO_NAME = args.video_name
VIDEO_LIST = args.video_list
if VIDEO_LIST:
    VIDEO_NAMES = []
    with open(VIDEO_NAME, 'r') as fid:
        for line in fid:
            VIDEO_NAMES.append(line.rstrip('\n'))
else:
    VIDEO_NAMES = [VIDEO_NAME]
USE_FLOW = args.use_flow
SAVE_JSON = args.save_json

GPU=args.gpu
models=[]
'''
models: list of tuples in the form of
        (model_proto, model_params, model_fusion_weight, input_type, conv_support, input_size).
        input_type is: 0-RGB, 1-Optical flow.
        conv_support indicates whether the network supports convolution testing, which is faster. If this is
                not supported, we will use oversampling instead
'''
models = [('models/resnet200_anet_2016_deploy.prototxt',
           'models/resnet200_anet_2016.caffemodel',
           1.0, 0, True, 224)]

if USE_FLOW:
    models.append(('models/bn_inception_anet_2016_temporal_deploy.prototxt',
                   'models/bn_inception_anet_2016_temporal.caffemodel.v5',
                   0.2, 1, False, 224))

cls = ActionClassifier(models, dev_id=GPU)

for EACH_VIDEO_NAME in VIDEO_NAMES:
    if SAVE_JSON:
        base_video = os.path.basename(EACH_VIDEO_NAME)
        if base_video.startswith('watch?v='):
            base_video = base_video[8:]
        out_file, _ = os.path.splitext(base_video)
        out_file += '.json'

        if os.path.exists(out_file):
            print "[Info] out_file={} already exists. Skipping...".format(
                    out_file)
            continue
        payload = dict()
        payload['top1_classification'] = dict()
        payload['top_classifications'] = dict()
        payload['class_probs'] = dict()
        payload['video_name'] = EACH_VIDEO_NAME

    try:
        rst = cls.classify(EACH_VIDEO_NAME)
    except:
        print "[Error] Classification on video={} failed. Skipping...".format(
                EACH_VIDEO_NAME
                )
        continue

    scores = rst[0]

    db = ANetDB.get_db("1.3")
    lb_list = db.get_ordered_label_list()
    idx = np.argsort(scores)[::-1]
    top_N = 3

    print '----------------Classification Results----------------------'
    for i in xrange(top_N):
        k = idx[i]
        label = lb_list[k]
        score = scores[k]
        print "\"{}\": {}".format(label, score)
        if SAVE_JSON:
            # np.floating -> float conversion is necessary
            payload['top_classifications'].update({label: float(score)})
            if i == 0:
                payload['top1_classification'].update({label: float(score)})

    if SAVE_JSON:
        for label, score in zip(lb_list, scores):
            payload['class_probs'][label] = float(score)
        #print "[Debug] type(payload)={}".format(type(payload))
        #print "[Debug] payload={}".format(payload)
        with open(out_file, 'w') as fid:
            json.dump(payload, fid)