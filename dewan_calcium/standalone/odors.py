def normalize_odors(odor_list, experiment_type, cell_class):
    normalized = []
    if experiment_type == 'Identity':
        if cell_class == 'VGAT':
            for odor in odor_list:
                if "2" in odor:
                    normalized.append(odor[1:])
                else:
                    normalized.append(odor)

    return normalized