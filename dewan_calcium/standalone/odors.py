def normalize_odors(odor_list, experiment_type, cell_class):
    normalized = []
    if experiment_type == 'Identity':
        if cell_class == 'VGAT':
            for odor in odor_list:
                normalized.append(if_two_remove(odor))

    elif experiment_type == 'Concentration':
        for odor in odor_list:
            _odor = if_sep_remove(odor, ['-', '_'])
            normalized.append(if_two_remove(_odor))

    return normalized

def if_two_remove(odor):
    if "2" in odor:
        return odor[1:]
    else:
        return odor

def if_sep_remove(odor, separators):
    for separator in separators:
        if separator in odor:
            odor_parts = odor.split(separator)
            try:
                _ = int(odor_parts[0])
                _odor = "".join([odor_parts[1], odor_parts[0]])
            except ValueError:
                _odor = "".join([odor_parts[0], odor_parts[1]])

            return _odor

    return odor
