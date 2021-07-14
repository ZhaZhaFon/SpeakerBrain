# handle data processing affairs

# requirements
import csv
import glob
import numpy as np
import os
import random
import re
import torch
from torch.utils.data import (
    RandomSampler,
    WeightedRandomSampler,
    DistributedSampler,
    Sampler,
)
import torchaudio
import tqdm

# internal code requirements
import sys
sys.path.append('/data/speakerbrain/speakerbrain')
import sysio

# enumerate VoxCeleb1 and make a csv for further fabricating the dataset
def prepare_voxceleb1(data_dir,
                      train_csv,
                      dev_csv,
                      trial_pairs,
                      split_ratio,
                      random_crop,
                      crop_duration):
    
    conf = {
        'data_dir': data_dir,
        'train_csv': train_csv,
        'dev_csv': dev_csv,
        'trial_pairs': trial_pairs,
        'split_ratio': split_ratio,
        'random_crop': random_crop,
        'crop_duration': crop_duration
    }

    # check if dataset directory is valid
    print('### 检查数据集目录')
    if os.path.exists(data_dir) == False:
        alert_msg = '!!! 数据集目录'+str(data_dir)+'不存在 !!!'
        raise AssertionError(alert_msg)
    if len(os.listdir(data_dir)) == 0:
        alert_msg = '!!! 数据集目录'+str(data_dir)+'为空 !!!'
        raise AssertionError(alert_msg)
    
    # split the dataset into train/dev
    print('### 数据集划分')
    train_list, dev_list = split_list(
        data_folder=data_dir,
        split_ratio=split_ratio,
        veri_file=os.path.join(data_dir, trial_pairs.split('/')[-1]),
        seg_dur=crop_duration
    )

    # make csv
    prepare_csv(train_list, train_csv, random_crop=random_crop, duration=crop_duration)
    prepare_csv(dev_list, dev_csv, random_crop=random_crop, duration=crop_duration)

# split the dataset into train/dev
def split_list(data_folder, split_ratio, veri_file, seg_dur):
    
    train_list = []
    dev_list = []
    test_list = []

    # testset trial list
    print('### 从数据集剥离被测说话人')
    # mark down all test samples in testset trial_pair
    test_list = [
        line.rstrip('\n').split(' ')[1]
        for line in open(veri_file)
    ] 
    test_list = set(sorted(test_list))
    test_spks = [snt.split('/')[0] for snt in test_list] # speaker id of testset sample

    path = os.path.join(data_folder, 'wav', '**', '*.wav') # assemble
    audio_files_dict = {}
    for f in glob.glob(path, recursive=True): # path match
        spk_id = f.split('/wav/')[1].split('/')[0] # get speaker id
        if spk_id not in test_spks:
            audio_files_dict.setdefault(spk_id, []).append(f)
    spk_id_list = list(audio_files_dict.keys())
    random.shuffle(spk_id_list) # shuffle

    print('### train/dev划分')
    split = int(0.01*split_ratio[0]*len(spk_id_list)) # split according to split_ratio
    for spk_id in spk_id_list[:split]: # trainset
        train_list.extend(audio_files_dict[spk_id])
    for spk_id in spk_id_list[split:]: # validset
        dev_list.extend(audio_files_dict[spk_id])

    return train_list, dev_list

# make csv
def prepare_csv(data_list, csv_path,  duration, random_crop=False):

    print(f'### {len(data_list)}条样本生成至{csv_path}')
    csv_output = [['ID', 'duration', 'wav', 'start', 'end', 'spk_id']]
    my_sep = '--'
    entry = []
    for wav_file in tqdm.tqdm(data_list, dynamic_ncols=True):
        try:
            [spk_id, sess_id, utt_id] = wav_file.split('/')[-3:]
        except ValueError:
            print(f'Malformed path: {wav_file}')
            continue
        audio_id = my_sep.join([spk_id, sess_id, utt_id.split('.')[0]])

        wav, sr = torchaudio.load(wav_file)
        wav = wav.squeeze(0)

        audio_duration = wav.shape[0]/sr
        if random_crop:
            uniq_chunk_list = get_chunks(duration, audio_id, audio_duration)
            for chunk in uniq_chunk_list:
                s, e = chunk.split('_')[-2:]
                start_sample = int(float(s)*sr)
                end_sample = int(float(e)*sr)
                # in case non-speech chunk is selected
                mean = torch.mean(np.abs(wav[start_sample:end_sample]))
                if mean < 5e-04:
                    continue
                csv_line = [
                    chunk,
                    str(duration),
                    wav_file,
                    start_sample,
                    end_sample,
                    spk_id,
                ]
        else:
            start_sample = 0
            stop_sample = wav.shape[0]

            csv_line = [
                audio_id,
                str(audio_duration),
                wav_file,
                start_sample,
                stop_sample,
                spk_id,
            ]
            entry.append(csv_line)
    csv_output = csv_output+entry
    with open(csv_path, mode='w') as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_output:
            csv_writer.writerow(line)

def get_chunks(duration, audio_id, audio_duration):

    num_chunks = int(audio_duration/duration)
    chunk_list = [
        audio_id+'_'+str(i*duration)+'_'+str(i*duration+duration)
        for i in range(num_chunks)
    ]

    return chunk_list