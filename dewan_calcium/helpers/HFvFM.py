import numpy as np

def get_pseudotrials(cell_trace_indices, PSEUDOTRIAL_LEN_S, ENDOSCOPE_FRAMERATE):
    pseudotrial_indices = []
def get_pseudotrial_pairs(trial_length, PSEUDOTRIAL_LEN_S, ENDOSCOPE_FRAMERATE):

    pseudotrial_length_frames = PSEUDOTRIAL_LEN_S * ENDOSCOPE_FRAMERATE

    frame_numbers = np.arange(0, trial_length+1, pseudotrial_length_frames)
    frame_pairs = list(zip(frame_numbers[:-1], frame_numbers[1:]))

    return frame_pairs

    return pseudotrial_indices

def get_trial_labels(num_trials, HF_first):
    import math
    trial_labels = []
    temp_num_trials = math.ceil(num_trials / 2)

    for i in range(temp_num_trials):
        if HF_first:
            trial_labels.append(f'HF-{i + 1}')
            trial_labels.append(f'FM-{i + 1}')
        else:
            trial_labels.append(f'FM-{i + 1}')
            trial_labels.append(f'HF-{i + 1}')

    if num_trials % 2 == 1:
        trial_labels = trial_labels[:-1]

    return trial_labels