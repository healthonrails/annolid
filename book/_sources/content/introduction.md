# Introduction
Annolid stands for **Annotation + Annelid** (segmentation).

Annolid is a toolkit for video-based research workflows that combines:
- Annotation (polygons, keypoints, and event labels)
- Instance segmentation and multi-object tracking
- Keypoint tracking and downstream analysis (e.g., place preference, motion/freezing metrics)

## What Annolid can do today
Annolid’s feature set evolves quickly, but the core workflows are stable:
- **Fast labeling in the GUI** (LabelMe-based), with manual tools plus AI-assisted tools.
- **AI-assisted polygons** using point prompts (Segment Anything family) and **text prompts** (Grounding DINO → SAM).
- **Video tracking backends** including Cutie / EfficientTAM-style VOS, plus model-driven segmentation/pose options (e.g., YOLO).
- **Export and interoperability**: LabelMe JSON, CSV summaries, COCO, and YOLO dataset conversion.
- **Behavior/event utilities**: event marking in the GUI, time-budget summaries, and post-hoc analyses.

If you need help or encounter an issue, please open an issue or use the community links in [Get in touch](get_in_touch).

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

Video courtesy of Rikki Laser and Alex Ophir:
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
  <iframe width="720" height="480" src="https://www.youtube.com/embed/qFABuhoGr_E" frameborder="0" allowfullscreen="true"> </iframe>
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
You can find these videos, tutorials on how to best use Annolid, as well as examples in Annolid's YouTube playlist [here](https://www.youtube.com/playlist?list=PLYp4D9Y-8_dRXPOtfGu48W5ENtfKn-Owc).
