import numpy as np

def get_pseudotrials(cell_trace_indices, PSEUDOTRIAL_LEN_S, ENDOSCOPE_FRAMERATE):
    pseudotrial_indices = []

    pseudotrial_length_frames = PSEUDOTRIAL_LEN_S * ENDOSCOPE_FRAMERATE

    for name, (start, end) in cell_trace_indices.iterrows():
        frame_numbers = np.arange(start, end+1, pseudotrial_length_frames)
        frame_pairs = list(zip(frame_numbers[:-1], frame_numbers[1:]+1))
        pseudotrial_indices.append(frame_pairs)

    return pseudotrial_indices