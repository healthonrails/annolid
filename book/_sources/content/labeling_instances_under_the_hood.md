
# Labeling instances\:  under the hood

## Labeling with Point Prompt

When you select the "AI Polygons" option, you can click points in the video frame or image. Annolid will then use efficientSAM by default to produce a polygon around the instance.

Please refer to this [video](https://youtu.be/YOnUnpwkLFc?si=bbmKXpPiGe2_G5h7) for a step-by-step guide.


## Labeling with text prompt
Annolid allows you to input text prompts with comma-separated labels, such as 'cat, dog', to utilize Grounding-DINO for generating bounding boxes. These bounding boxes will then serve as prompts for High-Quality SAM to generate masks, which will subsequently be converted into polygons.

Here is a video guiding you through the process: [Watch Video](https://youtu.be/bCCobNAwWvM?si=-mbTc2QfjnNv9FKy)
.
