import setuptools
from pathlib import Path

here = Path(__file__).parent

with open(here / "README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="annolid",
    version="1.0.1",
    author="Chen Yang",
    author_email="healthonrails@gmail.com",
    description="An Annotation and instance segmenation based mutiple animal tracking package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/healthonrails/annolid",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy>=1.18.2', 'scipy>=1.5.2', 'easydict>=1.9',
                      'opencv-python==4.1.2.30',
                      'opencv-contrib-python==4.1.2.30',
                      'opencv-python-headless==4.1.2.30',
                      'PyYAML>=5.3',
                      'labelme>=4.5.6',
                      'decord>=0.4.0',
                      'torch>=1.4.0',
                      'torchvision>=0.5.0',
                      'pandas>=1.1.3',
                      'pycocotools>=2.0.2',
                      'tensorboard>=2.3.0',
                      'PyQtWebEngine>=5.15.1',
                      "dataclasses>=0.6; python_version < '3.7'",
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
