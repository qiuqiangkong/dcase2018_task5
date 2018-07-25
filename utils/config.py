sample_rate = 16000
clip_duration = 10
window_size = 1024
overlap = 360   # So that there are 240 frames in an audio clip
seq_len = 240
mel_bins = 64
stereo_channels = 4

# fold_for_validation = 0     # Use the 0-th fold for validation

labels = ['absence', 'cooking', 'dishwashing', 'eating', 'other', 'social_activity', 'vacuum_cleaner', 'watching_tv', 'working']

lb_to_ix = {lb: ix for ix, lb in enumerate(labels)}
ix_to_lb = {ix: lb for ix, lb in enumerate(labels)}
