import json
import glob
from pathlib import Path


def fix_image_path(anno_dir):
    jsons = glob.glob(str(Path(anno_dir) / '*.json'))
    for js in jsons:
        with open(js, 'r') as f:
            print(js)
            data = json.load(f)
            data['imagePath'] = Path(data['imagePath']).name

            with open(js, 'w') as fw:
                json.dump(data, fw)
            print(data['imagePath'])


if __name__ == '__main__':
    anno_dir = '/path/to/video88'
    fix_image_path(anno_dir)
