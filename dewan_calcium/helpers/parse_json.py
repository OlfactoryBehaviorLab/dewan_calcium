import json
from pathlib import Path


def load_json_file(json_file_path: Path) -> dict:
    json_contents = {}

    try:  # Try to open the json file, if successful read the raw data into a dict
        with open(json_file_path, "r") as json_file:
            json_contents = json.loads(json_file.read())

    except FileNotFoundError as fnf_error:
        print("JSON session file not found!")
        raise fnf_error
    except json.JSONDecodeError as json_error:
        print("JSON session file is corrupt!")
        raise json_error

    return json_contents


def get_session_settings(json_file_path: Path) -> tuple:

    session_settings = load_json_file(json_file_path)

    microscope = session_settings['microscope']  # Get config values for the actual miniscope

    gain = microscope['gain']  # Recording Gain
    led_power = microscope['led']['exPower']  # LED Power
    fps = microscope['fps']['fps']
    focal_planes = get_focal_planes(microscope)  # Get the one, or multiple, focal planes

    return gain, led_power, fps, focal_planes


def get_focal_planes(microscope: dict) -> list:

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


def get_outline_coordinates(path: Path):
    outline_data = load_json_file(path)

    return outline_data
