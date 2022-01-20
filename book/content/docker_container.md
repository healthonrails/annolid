# Docker

We built a docker container for annolid to make it easier to access the package without the need to install anything beside Docker.

You need to make sure that Docker is installed on your system.
This has only been test on Ubuntu 20.04 LTS.

```
cd annolid/docker
docker build .
xhost +local:docker
docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix/ -e DISPLAY=$DISPLAY  <Image ID>
```
