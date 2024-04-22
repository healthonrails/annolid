# Introduction with examples
Annolid stands for:  Annotation + Annelid (segmentation).

Annolid is based on instance segmentation models. Instance segmentation is the task of attributing every pixel of an image to a specific category. It can be used to detect and delineate each distinct object of interest appearing in that image. As such it facilitates the tracking of multiple animals and along with it the flexible state identification (e.g., behavior classification, urine deposition, interactions among objects). Annolid has self-supervised, weakly-supervised, and unsupervised training options. We are striving to incorporate optical flow mechanics to improve performances as well as improving labeling efficiency via autolabeling and iterative model training.

Currently, Annolid is a work-in-progress, still in its alpha version, and subject to major changes. Nevertheless we hope you can use this jupyterbook as an efficient support to guide you through the process of using Annolid for your specific use case.

If you need help or encounter an issue don't hesitate to reach out to the developers by openning an [issue](get_in_touch) on Github.

## Video introduction
Below is a brief introduction to annolid:

<figure class="video_container">
  <iframe width="720" height="480" src="https://www.youtube.com/embed/tVIE6vG9Gao" frameborder="0" allowfullscreen="true"> </iframe>
</figure>

## Annolid can be applied to many diverse goals


- Animal Tracking
- Keypoints tracking (i.e. body parts)
- Automated behavior recognition

<figure class="video_container">
  <iframe width="720" height="480" src="https://www.youtube.com/embed/op3A4_LuVj8" frameborder="0" allowfullscreen="true"> </iframe>
</figure>


- Multiple animal tracking, including periods of partial body occlusion
- Whole-body masking
- Automated identification of interactions

Video courtesy of Caitlyn Finton and Alex Ophir:
<figure class="video_container">
  <iframe width="720" height="480" src="https://www.youtube.com/embed/dZ-qsD8uu-U" frameborder="0" allowfullscreen="true"> </iframe>
</figure>


- Masking and automatic scoring of lone animals and huddles of multiple animals
- Confidence of identification reported

Video courtesy of Rikki Laser and Alex OphirL:
<figure class="video_container">
  <iframe width="720" height="480" src="https://lh5.googleusercontent.com/FyOrtO6nEGeBEgEnZeuPf66cfqanl7NNmJFnHG7tJRnnvEOrf0FFfKNjT64pIS2HHjMs3queacFYFBVt4n18s4U1Dr6r7m3IYfEJzit83dh4UVRuUOpRUlU0UUjl0a7Bd6LACqGBuVc" frameborder="0" allowfullscreen="true"> </iframe>
</figure>


- Animal and object tracking, including periods of occlusion
- Tracked objects automatically associated with user-defined zones
- Robustness to noisy background

Video courtesy of Emily Sattora and Christiane Linster:
<figure class="video_container">
  <iframe width="720" height="480" src="https://lh6.googleusercontent.com/KhRfq5CfxTUvxVh-CBcQpW-hg1W83_-Hj3KSphdnfZJo_8oMZ5ED47erJ9mnFAqWBG7FphpzWN8qvM2sedAeuX-3NnqoJcDEBtAZtJlBQp1g3joyMX4vCdmxzqL_rM6D--wY-pJxtUo" frameborder="0" allowfullscreen="true"> </iframe>
</figure>


- Identification of freezing behavior  (e.g., from fear conditioning)
- Reporting of motion score based on optical flow measurements applied selectively to the body mask
<figure class="video_container">
  <iframe width="720" height="480" src="https://www.youtube.com/embed/qFABuhoGr_E
" frameborder="0" allowfullscreen="true"> </iframe>
</figure>


- Multiple animal tracking on cryptic background

Video courtesy of Jessica Nowicki, Julia Cora-anne Lee, and Lauren O’Connell:
<figure class="video_container">
  <iframe width="720" height="480" src="https://lh5.googleusercontent.com/CrYqegHOYWqkrYnbOHXSfK3n2T8iuyYHPyIQGBiMHwlltb6CIs4fO02KR9BAxz1Nju747gkKN32v5bEGlLDmpnQfCa5r8T9GSetMEtpp740D2KubY3f3rZCGdnbQxmqc92cuA7jmV7I" frameborder="0" allowfullscreen="true"> </iframe>
</figure>


- Multiple animal tracking with a large field of view

Video courtesy of Santiago Forero and Alex Ophir:
<figure class="video_container">
  <iframe width="720" height="480" src="https://drive.google.com/file/d/1cYdmueC-CaMhScpcB2E-eL9mkGDFuVCs/preview" frameborder="0" allowfullscreen="true"> </iframe>
</figure>


## Youtube playlist
You can find these videos, tutorials on how to best use Annolid as well as exemples in Annolid's youtube playlist [here](https://www.youtube.com/playlist?list=PLYp4D9Y-8_dRXPOtfGu48W5ENtfKn-Owc).
