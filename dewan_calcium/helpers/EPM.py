import pandas as pd


def find_led_start(points: pd.DataFrame) -> list:
    indexes = points.index[points['led_p'] > 0.98].values
    bins = []
    temp_bin = []
    for i in range(len(indexes) - 1):
        i1 = indexes[i]
        i2 = indexes[i + 1]

        diff = i2 - i1
        if not temp_bin:
            temp_bin.append(i1)
        else:
            if i == len(indexes) - 2:  # End is len - 1, and then -1 again for the zero indexes
                temp_bin.append(i2)
                bins.append(temp_bin)
                temp_bin = []
            elif diff == 1:
                continue
            elif diff > 1:
                temp_bin.append(i1)
                bins.append(temp_bin)
                temp_bin = []

    return bins
