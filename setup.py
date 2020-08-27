import setuptools

with open("README.md","r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="annolid",
    version="1.0.0",
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
    install_requires=['numpy','opencv-python','opencv-contrib-python'],
    python_requires='>=3.6',


)