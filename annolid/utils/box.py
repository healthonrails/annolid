"""box python API https://github.com/box/box-python-sdk
Please install  boxsdk
`pip install boxsdk`
"""
import json
from boxsdk import OAuth2, Client


def get_box_client(config_file='../config/box_config.json'):
    """authenticate with box.com return app client 
    to access and mananage box folder and files

    Args:
        config_file (str, optional): this file can be downloaded 
        from https://your_org_name.app.box.com/developers/console/app/xxxxxxx/configuration
        Defaults to '../config/box_config.json'.
    Please copy developer token from the above url and and a row `"developer_token":"YOUR_TOKEN",`
    to the box_config.json file.
    Returns: 
       Client: box client object
    """
    with open(config_file, 'r') as cfg:
        box_cfg = json.load(cfg)
    client_id = box_cfg['boxAppSettings']['clientID']
    client_secret = box_cfg['boxAppSettings']['clientSecret']
    token = box_cfg['developer_token']

    oauth = OAuth2(
        client_id=client_id,
        client_secret=client_secret,
        access_token=token,
    )
    client = Client(oauth)
    return client


def get_box_folder_items(client, folder_id='0'):
    """get box folder and items in side the folder with provide folder id

    Args:
        client (Client): box API client
        folder_id (str, optional): folder 
        ID, e.g. from app.box.com/folder/FOLDER_ID
        . Defaults to '0'.

    Returns:
        Folder, File: box folder and file objects
    """
    box_folder = client.folder(folder_id=folder_id)
    return box_folder, box_folder.get_items()


def upload_file(box_folder, local_file_path):
    """upload a local file to the box folder

    Args:
        box_folder (folder object)
        local_file_path (str): local file absolute path e.g. /data/video.mp4
    """
    box_folder.upload(local_file_path)


def download_file(box_file, local_file_path):
    """Download a file in box to local disk

    Args:
        box_file (File): box file object
        local_file_path (str): local file path e.g. /data/video.mp4
    """
    with open(local_file_path, 'wb') as lf:
        box_file.download_to(lf)


def is_results_complete(box_folder,
                        result_file_pattern='_motion.csv',
                        num_expected_results=0
                        ):
    """Check if a box folder contains all the expected result files

    Args:
        box_folder (BoxFolder): box folder object
        result_file_pattern (str, optional): pattern in the file. Defaults to '_motion.csv'.

    Returns:
        bool: True if the folder contails the expected number of result files else False
    """
    num_of_results = 0
    for bf in box_folder.get_items():
        if result_file_pattern in bf.name:
            num_of_results += 1
    return num_of_results == num_expected_results
