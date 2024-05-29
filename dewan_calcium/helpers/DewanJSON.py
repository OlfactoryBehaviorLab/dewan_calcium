import json
import numpy as np

from pathlib import Path
import os


def get_json_settings(json_file_path: Path) -> list:

    session_settings = load_json_file(json_file_path)


def load_json_file(json_file_path: Path) -> dict:
    session_settings = {}

    try:  # Try to open the session.json file, if successful read the raw data into a dict
        with open(json_file_path, "r") as json_file:
            session_settings = json.loads(json_file.read())

    except FileNotFoundError as fnf_error:
        print("JSON session file not found!")
        raise fnf_error
    except json.JSONDecodeError as json_error:
        print("JSON session file is corrupt!")
        raise json_error

    return session_settings


def get_focal_planes(session_file: Path) -> list:

    session_settings = load_json_file(session_file)

    microscope = session_settings['microscope']
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