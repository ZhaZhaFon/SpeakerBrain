# main program for training

# system requirements
import argparse

# file & os requirements
from hyperpyyaml import load_hyperpyyaml
import os

# deep learning requirements
import torch

# speech processing requirements 
import soundfile as sf

# internal code requirements
import sys
sys.path.append('/data/speakerbrain/speakerbrain')
import base
import data
import sysio

# argument parser
parser = argparse.ArgumentParser(description='### VoxCeleb-SpeakerRecognition的argument parser')
# path to load yaml for hyper-parameter
parser.add_argument('-y', '--yaml', type=str, default='./hparams_xvector.yaml')
# whether to continue training from a checkpoint
parser.add_argument('-c', '--continue', type=bool, default=False)

# main function
if __name__ == '__main__':
    
    # argument parser
    args = dict(vars(parser.parse_args()))

    # load yaml file
    print('# 解析yaml...')
    with open(args['yaml']) as f:
        hparams = load_hyperpyyaml(f)

    # create a directory for thiscurrent experiment
    print('# 保存实验...')
    sysio.make_dir(hparams['save_path'])

    # prepare a csv for the whold dataset
    print('# 数据准备...')
    base.run_on_main(
        data.prepare_voxceleb1,
        kwargs={
            'data_dir': hparams['data_dir'],
            'train_csv': hparams['train_csv'],
            'dev_csv': hparams['dev_csv'],
            'trial_pairs': hparams['trial_pairs'],
            'split_ratio': hparams['split_ratio'],
            'random_crop': hparams['random_crop'],
            'crop_duration': hparams['crop_duration']
        }
    )