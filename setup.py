#!/usr/bin/env python

import setuptools

setuptools.setup(
    name="annolid",
    version="1.2.1",
    author="Chen Yang",
    author_email="healthonrails@gmail.com",
    description="An annotation and instance segmentation-based multiple animal tracking and behavior analysis package.",
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
                      'labelme==5.5.0',
                      'decord==0.4.0; platform_system!="Darwin"',  # decord for non-macOS platforms
                      'eva-decord>=0.6.1; sys_platform=="darwin"',  # eva-decord for macOS
                      'sounddevice>=0.4.6',
                      'librosa>=0.10.1',
                      'torch>=1.7',
                      'torchvision>=0.8',
                      'pandas>=1.1.3',
                      'pycocotools>=2.0.2',
                      'tensorboard>=2.3.0',
                      'imageio>=2.8.0',
                      'shapely>=2.0.2',
                      'imageio-ffmpeg>=0.4.2',
                      'qimage2ndarray>=1.8',
                      'simplification==0.6.11',
                      'tokenizers',
                      "dataclasses>=0.6; python_version < '3.7'",
                      'einops >= 0.6',
                      'omegaconf >= 2.3.0',
                      'hydra-core >= 1.3.2',
                      'timm >=0.9.7',
                      'h5py >=3.10.0',
                      'psutil >=5.9.8',
                      'pyserial >= 3.5',
                      # 'PyQt5 == 5.15.7',
                      'Pillow>=9.3.0,<=9.5.0',
                      "chardet>=5.2.0",
                      "transformers>=4.45.1",
                      "scikit-learn-extra >= 0.3.0",
                      'ultralytics>=8.3.34',
                      # "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git",
                      "segment-anything @ git+https://github.com/SysCV/sam-hq.git",
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
