#!/usr/bin/env python

import setuptools

setuptools.setup(
    name="annolid",
    version="1.1.3",
    author="Chen Yang",
    author_email="healthonrails@gmail.com",
    description="An Annotation and instance segmenation based mutiple animal tracking package",
    url="https://github.com/healthonrails/annolid",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy>=1.18.2',
                      'scipy>=1.5.2',
                      'easydict>=1.9',
                      'opencv-contrib-python>=4.1.2.30',
                      'opencv-python-headless>=4.1.2.30',
                      'PyYAML>=5.3',
                      'labelme==5.3.1',
                      #'decord>=0.4.0',
                      'torch>=1.7',
                      'torchvision>=0.8',
                      'pandas>=1.1.3',
                      'pycocotools>=2.0.2',
                      'tensorboard>=2.3.0',
                      'imageio>=2.8.0',
                      'imageio-ffmpeg>=0.4.2',
                      'qimage2ndarray>=1.8',
                      'simplification>=0.5.7',
                      "dataclasses>=0.6; python_version < '3.7'",
                      #"segment-anything @ git+https://github.com/facebookresearch/segment-anything.git",
                      ],
    python_requires='>=3.6',

    entry_points={
        'console_scripts': [
            'annolid = annolid.gui.app:main',
            'annolid-train = annolid.segmentation.yolact.train:train',
            'annolid-track = annolid.segmentation.yolact.eval:main',
        ],
    },


)
