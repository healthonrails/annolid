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
