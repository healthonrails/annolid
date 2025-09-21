# Track animals and Auto labeling

Click `Track Animals` button on the toolbar, fill the info in the opened dialog as follows. 

![Track Animals and objects](../imgs/track_animals.png)

Use `Detectron2` as the default model type

Choose the video file path and provide the trained model file with .pth format

Select a class threshold between 0 and 1

Provide the data.yaml file path in the COCO dataset folder

The output result folder is optional.

Note. You need to [install `Detectron2`](https://healthonrails.github.io/annolid/install.html#install-detectron2-locally) on your local device. If your workstation does not have a GPU card, it will only extract the key frames from the provided video and will save predicted results as json format in the same png image folder. 
Here is an example of predicted polygon annotions. 

![](../imgs/predicted_polygons.png)
The GPU workstation will run inference for all the frames in the provided video and will save the predicted results into a CSV file.

# Output CSV format

Here are the columns of the Annolid CSV output format: 

frame_number: int, 0 based numbers for frames e.g. 10 the 11th frame

x1: float, the top left x value of the instance bounding box

y1: float, the top left y value of the instance bounding box

x2: float, the bottom right x value of the instance bounding box

y2: float, the bottom right y value of the instance bounding box

instance_name: string, the unique name of the instances or the class name

class_score: float, the confidence score between 0 to 1 for the class or instance name

segmentation: run length encoding of the instance binary mask

cx: float, optional, the center x value of the instance bounding box

cy: float, optional, the center y value of the instance bounding box


# Cutie + DINO Body-Part Tracker

The Cutie + DINO tracker pairs Cutie's video object segmentation with DINO patch descriptors to preserve keypoint identity across long and challenging clips. The runtime can be tweaked through `CutieDinoTrackerConfig` to prioritise appearance, structure, or symmetry cues depending on the animal and camera setup.

## Key runtime parameters

- `appearance_bundle_radius` / `appearance_bundle_size` / `appearance_bundle_weight` control the per-keypoint appearance codebook. Increase the radius and size to gather more context in the first frame; raise the weight when fur patterns are distinctive and you want candidate patches that match the stored descriptors to win.
- `baseline_similarity_weight` penalises candidates that deviate from the initial (first-frame) descriptor quality. Increase it to keep keypoints glued to their original look when lighting is stable; lower it when the appearance changes dramatically over time.
- `structural_consistency_weight` keeps the inter-part distances close to the first frame. Boost it for rigid animals or cameras with minimal perspective distortion, and reduce it for highly articulated limbs.
- `symmetry_pairs` / `symmetry_penalty` define left/right identities that must not flip. List each `(left_label, right_label)` pair that should stay mirrored (labels match the annotation dialog). Increase the penalty for fast-moving symmetric animals like mice so cross-over candidates are heavily discouraged.
- `max_candidate_tracks` limits how many patch matches are considered per keypoint each frame. Raise it if the animal moves rapidly and you need a wider search; drop it for performance when motion is smooth.
- `mask_dilation_iterations`, `mask_dilation_kernel`, `mask_similarity_bonus`, and `max_mask_fallback_frames` control the mask-aware matching stage. They determine how aggressively stored masks are dilated and how much bonus a candidate receives for staying inside the expected region when Cutie has fresh or fallback predictions.
- `support_probe_count` / `support_probe_sigma` / `support_probe_radius` / `support_probe_weight` sample Gaussian-distributed “support probes” around each keypoint to keep local context consistent. Increase the count or weight when nearby appearance cues (fur tufts, tattoos) help disambiguate swaps; widen the radius if the animal stretches markedly between frames. `support_probe_mask_only` restricts probes to Cutie’s mask and `support_probe_mask_bonus` adds a small reward when probes stay inside the current mask corridor.

## Tuning guidance

- **Symmetric rodents during sharp turns:** set `symmetry_pairs` for all left/right body parts and bump `symmetry_penalty` to `0.6–0.8`. Keep `structural_consistency_weight` at `0.3+` so ear and tail distances stay coherent. This prevents swaps when the animal spins.
- **Occlusions or Cutie drop-outs:** allow at least `2` fallback frames and enlarge the dilation kernel to `3` so the tracker can keep keypoints anchored while masks recover. Raising `mask_similarity_bonus` (e.g. `0.4–0.5`) helps the assignment engine prefer in-mask candidates even when descriptor similarity is ambiguous.
- **Low-texture animals:** increase `appearance_bundle_radius` to gather more context and push `appearance_bundle_weight` toward `0.4`. Pair it with a modest `baseline_similarity_weight` (<`0.2`) to avoid punishing lighting shifts.
- **High-speed motion:** lift `max_candidate_tracks` to `10` and relax `baseline_similarity_weight`. Consider lowering `velocity_smoothing` so the velocity estimate reacts quickly to direction changes.
- **Ambiguous local descriptors:** raise `support_probe_weight` toward `0.5` and add more probes so the tracker checks neighbouring fur/texture consistency before accepting a candidate. If the mask is reliable, keep `support_probe_mask_only=True` to bias probes toward valid anatomy.
- **Quality monitoring:** each JSON keypoint now carries `velocity`, `misses`, and `quality` flags. Downstream analytics can watch for spikes in `misses` or drops in `quality` to trigger re-initialisation or manual review.
