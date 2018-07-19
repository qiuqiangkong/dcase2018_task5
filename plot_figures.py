import argparse
import matplotlib.pyplot as plt
import os

from features import LogMelExtractor, calculate_logmel, read_meta
import config


def plot_logmel(args):
    """Plot log Mel feature of one audio per class. 
    """

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    
    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    seq_len = config.seq_len
    mel_bins = config.mel_bins
    labels = config.labels
    classes_num = len(labels)
    
    plot_num = 12
    
    # Paths
    meta_csv = os.path.join(dataset_dir, 'meta.txt')
    audios_dir = os.path.join(dataset_dir, 'audio')
    
    # Feature extractor
    feature_extractor = LogMelExtractor(sample_rate=sample_rate, 
                                        window_size=window_size, 
                                        overlap=overlap, 
                                        mel_bins=mel_bins)
    
    

    # Calculate log mel feature of audio clips
    (audio_names, target_labels, sessions) = read_meta(meta_csv)
    
    features_list = []
    selected_audio_names = []
    
    # Select one audio per class and extract feature
    for label in labels:
        
        for (n, audio_name) in enumerate(audio_names):
        
            if label == target_labels[n]:
                
                audio_path = os.path.join(audios_dir, audio_name)
                
                feature = calculate_logmel(audio_path=audio_path, 
                                           sample_rate=sample_rate, 
                                           feature_extractor=feature_extractor)
                     
                features_list.append(feature)
                selected_audio_names.append(audio_name)
                                        
                break
        
    # Plot
    rows_num = 3
    cols_num = 3
    n = 0
    
    fig, axs = plt.subplots(rows_num, cols_num, figsize=(10, 6))
    
    for n in range(classes_num):
        row = n // cols_num
        col = n % cols_num
        axs[row, col].matshow(features_list[n].T, origin='lower', aspect='auto', cmap='jet')
        axs[row, col].set_title('No. {}, {}'.format(n, labels[n]))
        axs[row, col].set_ylabel('log Mel')
        axs[row, col].yaxis.set_ticks([])
        axs[row, col].xaxis.set_ticks([0, seq_len])
        axs[row, col].xaxis.set_ticklabels(['0', '10 s'], fontsize='small')
        axs[row, col].xaxis.tick_bottom()
    
    for n in range(classes_num, rows_num * cols_num):
        row = n // cols_num
        col = n % cols_num
        axs[row, col].set_visible(False)
    
    for n in range(classes_num):
        print('No. {}, {}'.format(n, selected_audio_names[n]))
    
    fig.tight_layout()
    plt.show()
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_plot_logmel = subparsers.add_parser('plot_logmel')
    parser_plot_logmel.add_argument('--dataset_dir', type=str)
    
    args = parser.parse_args()
    
    if args.mode == 'plot_logmel':
        plot_logmel(args)
        
    else:
        raise Exception("Incorrect arguments!")