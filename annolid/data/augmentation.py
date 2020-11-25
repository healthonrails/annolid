import glob
import json
from pathlib import Path
import pandas as pd
import collections


class LabelStats:

    def __init__(self, anno_dir):
        self.anno_dir = anno_dir
        self.instance_counter = collections.defaultdict(int)

    def count(self):
        label_files = Path(self.anno_dir).glob("*.json")
        for lf in label_files:
            label_file = json.loads(lf.read_bytes())
            for shape in label_file['shapes']:
                label = shape['label']
                self.instance_counter[label] += 1
        return self.instance_counter

    def to_table(self):
        count_dict = self.count()
        df = pd.DataFrame(
            dict(count_dict).items(),
            columns=['instance_name', 'counts'
                     ]
        )
        return df.sort_values(by="counts")
