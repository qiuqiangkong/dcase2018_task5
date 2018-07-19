import numpy as np
import os
import pandas as pd
import argparse
import h5py
import librosa
from scipy import signal
import matplotlib.pyplot as plt
import time
import csv
import random

from utilities import read_stereo_audio, create_folder
import config


class LogMelExtractor():
    def __init__(self, sample_rate, window_size, overlap, mel_bins):
        
        self.window_size = window_size
        self.overlap = overlap
        self.ham_win = np.hamming(window_size)
        
        self.melW = librosa.filters.mel(sr=sample_rate, 
                                        n_fft=window_size, 
                                        n_mels=mel_bins, 
                                        fmin=20., 
                                        fmax=sample_rate // 2).T
    
    def transform(self, audio):
    
        ham_win = self.ham_win
        window_size = self.window_size
        overlap = self.overlap
    
        [f, t, x] = signal.spectral.spectrogram(
                        audio, 
                        window=ham_win,
                        nperseg=window_size, 
                        noverlap=overlap, 
                        detrend=False, 
                        return_onesided=True, 
                        mode='magnitude') 
        x = x.T
            
        x = np.dot(x, self.melW)
        x = np.log(x + 1e-8)
        x = x.astype(np.float32)
        
        return x


def read_meta(meta_csv):
    """Read meta csv. 
    
    Args:
      meta_csv: string, path of csv file
      
    Returns:
      (audio_names, labels, sessions) | (audio_names,)
    """
    
    with open(meta_csv, 'r') as f:
        
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)
        
        if len(lis[0]) == 1:
            audio_names = []
            data_type = 'test'
            
        elif len(lis[0]) == 3:
            audio_names = []
            labels = []
            sessions = []
            data_type = 'development'
            
        else:
            raise Exception('Incorrect meta!')
        
        
        for li in lis:
            
            if data_type == 'test':
                
                filename = li[0]
                audio_name = filename.split('/')[1]
                
                audio_names.append(audio_name)
            
            elif data_type == 'development':
                
                [filename, label, session] = li
                audio_name = filename.split('/')[1]
                
                audio_names.append(audio_name)
                labels.append(label)
                sessions.append(session)
                
        if data_type == 'development':
            return audio_names, labels, sessions
            
        elif data_type == 'test':
            return (audio_names,)
                

def checkfiles(args):
    """Check if none of audios are corrupted. 
    """
    
    # Arguments & parameters
    dataset_dir = args.dataset_dir
    
    sample_rate = config.sample_rate
    clip_duration = config.clip_duration
    stereo_channels = config.stereo_channels
    
    # Paths
    audios_dir = os.path.join(dataset_dir, 'audio')
    meta_csv = os.path.join(dataset_dir, 'meta.txt')
    
    # Read meta csv
    return_tuple = read_meta(meta_csv)
    
    if len(return_tuple) == 1:
        (audio_names,) = return_tuple
        data_type = 'test'
        
    elif len(return_tuple) == 3:
        (audio_names, labels, sessions) = return_tuple
        data_type = 'development'
    
    
    for (n, audio_name) in enumerate(audio_names):

        if n % 100 == 0:
            print(n)
        
        audio_path = os.path.join(audios_dir, audio_name)
        
        (stereo_audio, fs) = read_stereo_audio(audio_path, 
                                            target_fs=None, 
                                            to_mono=False)
                                            
        # Check audio
        corrupted = False
        
        if (fs != sample_rate) or \
            (stereo_audio.shape[0] != stereo_channels) or \
            (stereo_audio.shape[1] != sample_rate * clip_duration):
            
            corrupted = True
    
        if corrupted:
            print("{} is corrupted!".format(audio_name))
            
    print("Checkfiles finished! You may extract feature now!")


def calculate_logmel(audio_path, sample_rate, feature_extractor):
    
    # Read audio
    (mono_audio, fs) = read_stereo_audio(audio_path, 
                                         target_fs=sample_rate, 
                                         to_mono=True)
    '''(samples_num,)'''
    
    # Normalize energy
    mono_audio /= np.max(np.abs(mono_audio))
    
    # Extract feature
    feature = feature_extractor.transform(mono_audio)
    
    return feature


def logmel(args):

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    mini_data = args.mini_data
    
    sample_rate = config.sample_rate
    clip_duration = config.clip_duration
    window_size = config.window_size
    overlap = config.overlap
    seq_len = config.seq_len
    mel_bins = config.mel_bins
    stereo_channels = config.stereo_channels
    
    # Paths
    audios_dir = os.path.join(dataset_dir, 'audio')
    meta_csv = os.path.join(dataset_dir, 'meta.txt')
    
    return_tuple = read_meta(meta_csv)
    
    if len(return_tuple) == 1:
        (audio_names,) = return_tuple
        data_type = 'test'
        
    elif len(return_tuple) == 3:
        (audio_names, labels, sessions) = return_tuple
        data_type = 'development'
    
    if mini_data:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', 
                                 'mini_{}.h5'.format(data_type))
    else:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', 
                                 '{}.h5'.format(data_type))
    
    create_folder(os.path.dirname(hdf5_path))
    
    # Only use partial data when set mini_data to True
    if mini_data:
        
        audios_num = 300
        random_state = np.random.RandomState(0)
        audio_indexes = np.arange(len(audio_names))
        random_state.shuffle(audio_indexes)
        audio_indexes = audio_indexes[0 : audios_num]
        
        audio_names = [audio_names[idx] for idx in audio_indexes]
        
        if data_type == 'development':
            labels = [labels[idx] for idx in audio_indexes]
            sessions = [sessions[idx] for idx in audio_indexes]
    
    print("Number of audios: {}".format(len(audio_names)))
    
    # Feature extractor
    feature_extractor = LogMelExtractor(sample_rate=sample_rate, 
                                        window_size=window_size, 
                                        overlap=overlap, 
                                        mel_bins=mel_bins)

    # Create hdf5 file
    begin_time = time.time()
    hf = h5py.File(hdf5_path, 'w')
    
    hf.create_dataset(
        name='feature', 
        shape=(0, seq_len, mel_bins), 
        maxshape=(None, seq_len, mel_bins), 
        dtype=np.float32)
    
    # Calculate and write features
    for (n, audio_name) in enumerate(audio_names):
        
        print(n, audio_name)
        
        audio_path = os.path.join(audios_dir, audio_name)
        
        # Extract feature
        feature = calculate_logmel(audio_path=audio_path, 
                                   sample_rate=sample_rate, 
                                   feature_extractor=feature_extractor)
        '''(seq_len, mel_bins)'''
        
        print(feature.shape)
        
        hf['feature'].resize((n + 1, seq_len, mel_bins))
        hf['feature'][n] = feature
        
        # Plot log Mel for debug
        if False:
            plt.matshow(feature.T, origin='lower', aspect='auto', cmap='jet')
            plt.show()
    
    # Write out meta to hdf5 file
    hf.create_dataset(name='audio_name', 
                    data=[s.encode() for s in audio_names], 
                    dtype='S50')
        
    if data_type == 'development':
        
        hf.create_dataset(name='label', 
                        data=[s.encode() for s in labels], 
                        dtype='S20')
                          
        hf.create_dataset(name='session', 
                          data=[s.encode() for s in sessions], 
                          dtype='S10')

    hf.close()
                
    print("Write out to {}".format(hdf5_path))
    print("Time: {} s".format(time.time() - begin_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    parser_checkfiles = subparsers.add_parser('checkfiles')
    parser_checkfiles.add_argument('--dataset_dir', type=str, required=True)

    parser_logmel = subparsers.add_parser('logmel')
    parser_logmel.add_argument('--dataset_dir', type=str, required=True)
    parser_logmel.add_argument('--workspace', type=str, required=True)
    parser_logmel.add_argument('--mini_data', action='store_true', default=False)
    
    args = parser.parse_args()
    
    if args.mode == 'checkfiles':
        checkfiles(args)
    
    elif args.mode == 'logmel':
        logmel(args)

    else:
        raise Exception("Incorrect arguments!")
        