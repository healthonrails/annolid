import imageio
import glob
import json
from pathlib import Path
import pandas as pd
import collections
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


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


class Augmentation(LabelStats):

    def __init__(self, anno_dir):
        ia.seed(4)
        self.anno_dir = anno_dir
        super(Augmentation, self).__init__(anno_dir)
        self.df = self.to_table()
        self.augment_list = []

    def augment(self):
        few_instances = self.df[self.df['counts'] < self.df['counts'].median()]
        aug_list = list(few_instances['instance_name'])
        label_files = Path(self.anno_dir).glob("*.json")

        aug = iaa.Sequential([
            iaa.AdditiveGaussianNoise(scale=10),
            # The following transformations will change the polygon
            # iaa.Affine(rotate=(-0.05, 0.05), translate_percent=(-0.05, 0.05), scale=(0.8, 1.2),
            #            mode=["constant", "edge"], cval=0),
            # iaa.CoarseDropout(0.1,size_px=8),
            # iaa.Fliplr(0.5),
            #iaa.PerspectiveTransform((0.01, 0.01)),
            #iaa.LinearContrast((0.8, 1.2), per_channel=0.5),
            iaa.Sometimes(0.05, iaa.Snowflakes()),

            iaa.AddToHueAndSaturation((-50, 50)),
        ])

        for lf in label_files:
            label_file = json.loads(lf.read_bytes())
            img_path = lf.with_suffix('.jpg')
            img = imageio.imread(img_path)
            image_polys = np.copy(img)
            polys = []
            is_aug = False

            aug_dir = img_path.parent.parent / (img_path.parent.stem + '_aug')
            aug_dir.mkdir(exist_ok=True)

            for i, shape in enumerate(label_file['shapes']):
                label = shape['label']
                if label in aug_list:
                    is_aug = True
                    points = shape['points']

                    polygon = Polygon(points, [label])
                    psoi = ia.PolygonsOnImage(
                        [polygon], shape=image_polys.shape)
                    _, psoi_aug = aug(image=image_polys, polygons=psoi)
                    aug_points = psoi_aug.polygons[0].exterior
                    polys.append(polygon)
                    label_file['shapes'][i]['points'] = aug_points.tolist()

                    self.augment_list.append(lf)

            if is_aug:
                posi = ia.PolygonsOnImage(polys, shape=image_polys.shape)
                img_aug, psoi_aug = aug(image=image_polys, polygons=posi)
                aug_img_path = aug_dir / (img_path.stem + '_aug.jpg')
                aug_json_path = aug_img_path.with_suffix('.json')
                imageio.imsave(aug_img_path, img_aug, '.jpg')
                label_file["imageData"] = None
                label_file['imagePath'] = aug_img_path.name
                with open(aug_json_path, "w") as f:
                    json.dump(label_file, f, ensure_ascii=False, indent=2)
                ia.imshow(psoi_aug.draw_on_image(
                    img_aug, alpha_face=0.3, size_points=7))
        return set(self.augment_list)
