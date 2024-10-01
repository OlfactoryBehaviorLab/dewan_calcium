import numpy as np
import sys

if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else: from tqdm import tqdm

def get_pseudotrial_pairs(trial_length, PSEUDOTRIAL_LEN_S, ENDOSCOPE_FRAMERATE):

    pseudotrial_length_frames = PSEUDOTRIAL_LEN_S * ENDOSCOPE_FRAMERATE

    frame_numbers = np.arange(0, trial_length+1, pseudotrial_length_frames)
    frame_pairs = list(zip(frame_numbers[:-1], frame_numbers[1:]))

    return frame_pairs

def get_dff_for_pseudotrials(combined_data, cell_names, trial_labels, PSEUDOTRIAL_LEN_S, ENDOSCOPE_FRAMERATE):
    # This is a monstrosity, I'm sorry.
    pseudotrial_dff_per_cell = {}

    for cell in tqdm(cell_names, desc='Cell: '):  # Loop through each cell
        trial_dff_pseudotrials = {}
        cell_dff_data = combined_data[cell]

        for trial in trial_labels:  # Loop through each trial
            pseudotrial_dff = []
            trial_dff_values = cell_dff_data[trial]
            pseudotrial_pairs = get_pseudotrial_pairs(len(trial_dff_values), PSEUDOTRIAL_LEN_S,
                                                            ENDOSCOPE_FRAMERATE)
            # Get the pseudotrial pairings for this trial

            for pair in pseudotrial_pairs:  # Loop through each pair
                pseudotrial_dff_values = trial_dff_values.iloc[pair[0]:pair[1]].values  # Get df/F Values

                if not np.any(np.isnan(
                        pseudotrial_dff_values)):  # If the trial is cut short for any reason, we'll skip rows with nan
                    pseudotrial_dff.append(pseudotrial_dff_values)

            trial_dff_pseudotrials[trial] = pseudotrial_dff
        pseudotrial_dff_per_cell[cell] = trial_dff_pseudotrials

    return pseudotrial_dff_per_cell

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