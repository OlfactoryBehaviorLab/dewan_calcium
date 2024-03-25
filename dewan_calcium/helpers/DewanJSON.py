import json
import numpy as np

from pathlib import Path
import os


def get_focal_planes(session_file: Path) -> list:



    with open(session_file,'r') as j:
        inscopix_json = json.loads(j.read())

    microscope = inscopix_json['microscope']
    focal_plane = microscope['focus']

    multiplane_enabled = microscope['multiplane']['enabled']

    if multiplane_enabled:
        focal_planes = []
        planes = [key for key in microscope['multiplane']['planes'].keys()]

        for each in planes:
            plane = microscope['multiplane']['planes'][each]
            if plane['enabled']:
                focal_planes.append(plane['focus'])

        return focal_planes

    else:
        return focal_plane


def get_outline_coordinates(path: os.PathLike):

    try:
        json_file = open(path)
    except (FileNotFoundError, IOError):
        print("Error loading Cell_Contours File!")
        return None

    json_data = json.load(json_file)
    keys = np.array(list(json_data.keys()))

    json_file.close()

    return keys, json_data