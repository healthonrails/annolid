# Docker

We provide a script to build a docker container for Annolid to make it easier to access the package without the need to install anything besides Docker.

You need to make sure that [Docker](https://docs.docker.com/engine/install/ubuntu/) is installed on your system (or a similar software capable of building containerized applications)


```{note}
Currently this has only been tested on Ubuntu 20.04 LTS.
```


```
cd annolid/docker
docker build .
```

Now if you want to access Annolid through the GUI, you will need to connect the image through your computer's display using

```
xhost +local:docker
```

Finally, we will need the Image ID of your Annolid image. To get that use: 

```
docker image ls
```

and write down the IMAGE ID associated with annolid repository. 

Finally to launch annolid, run the following:

```
docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix/ -e DISPLAY=$DISPLAY <Image ID>
```
