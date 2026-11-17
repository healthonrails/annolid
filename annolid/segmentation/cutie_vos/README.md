# Putting the Object Back into Video Object Segmentation
Ho Kei Cheng, Seoung Wug Oh, Brian Price, Joon-Young Lee, Alexander Schwing

References:

```
@inproceedings{cheng2023putting,
  title={Putting the Object Back into Video Object Segmentation},
  author={Cheng, Ho Kei and Oh, Seoung Wug and Price, Brian and Lee, Joon-Young and Schwing, Alexander},
  booktitle={arXiv},
  year={2023}
}
```

https://github.com/hkchengrex/Cutie/tree/main

## Annolid Checkpoints

- `weights/cutie-base-mega.pth` & `weights/cutie-sam3-bodyparts.pth`: Directly compressed (`tucker_bn_folded_v2`) and Dual-Teacher (`SAM 3.1` + `DINOv3 ViT-L/16`) distilled `CUTIE` checkpoint (`15.76M` parameters, `30.26 MB`, `-55.0%` parameters, `4.43x` smaller disk footprint, `1.48x` faster propagation). `CUTIE.load_weights` remains 100% backward-compatible with legacy unfactorized checkpoints.
- `weights/cutie-base-mega.orig.pth`: Immutable backup of the untouched upstream Cutie Base Mega checkpoint (`MD5 a6071de6136982e396851903ab4c083a`).
