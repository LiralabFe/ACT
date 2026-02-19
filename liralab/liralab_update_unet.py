import tensorflow as tf
import re
import os

import h5py
import sys
import json
import argparse
from keras.models import model_from_config
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

INPUT_MODEL_PATH = "./segmentation_models/unet_dnet121_case_v1.h5"
OUTPUT_MODEL_PATH = "./segmentation_models/unet_dnet121_case_v2.h5"

sys.argv = ['program', '-o', './segmentation_models/', 
 '-m', './segmentation_models/unet_dnet121_case_v1.h5']


parser = argparse.ArgumentParser()
parser.add_argument('-m','--models', nargs='+', type=Path, help='List of model files/directories to fix.')
parser.add_argument('-x','--ext', default='h5', help='Model extension (if using directories)')
parser.add_argument('-o','--output', type=Path, help='Output directory.')
args = parser.parse_args()

for p in args.models:
    models = []
    if p.is_dir():
        models.extend([_ for _ in p.glob('*.{}'.format(args.ext))])
    else:
        if p.exists():
            models.append(p)
        else:
            print('Missing file: {}'.format(p))

args.models = models


def fix_model_file(fp,od=Path('.')):
    assert fp.exists()
    if not od.is_dir():
        od.mkdir(parents=True, exist_ok=True)
    op = str(od / fp.name)
    fp = str(fp)
    
    with h5py.File(fp) as h5:
        config = json.loads(h5.attrs.get("model_config")
                            .decode('utf-8')
                            .replace('input_dtype','dtype'))
    with tf.Session('') as sess:
        model = model_from_config(config)
        model.load_weights(fp)
        model.save(op)
        del model
    del sess
    print(op)

if __name__ == '__main__':
    for model in args.models:
        fix_model_file(model, od=args.output)