# Combining models

Annolid lets you mix and match tools depending on your data and the question you’re asking. A common pattern is:
1. Use a **foundation model** to reduce manual labeling.
2. Use a **tracker** to propagate labels through time.
3. Use a **task-specific model** (e.g., YOLO pose) if you need extra signals.
4. Export everything to CSV and combine analyses offline.

## Practical combinations that work well
### AI polygons + tracking
- Use **AI polygons** (SAM-family) to outline each instance on a single frame.
- Track forward with **Cutie / EfficientTAM-style** VOS backends.
- Review/correct, then export a tracking CSV.

### Text prompt + refinement
- Use **text prompts** (Grounding DINO → SAM) to quickly bootstrap object masks when manual clicks are slow.
- Convert masks to polygons and continue with your usual tracking workflow.

### Segmentation + keypoint tracking
- Use mask tracking (polygons) for identity and body region.
- Use DINO-based keypoint tracking (see `book/tutorials/DINOv3_keypoint_tracking.md`) when you need body-part trajectories.

### Custom YOLO + Annolid review
- Train a YOLO segmentation/pose model on your own labels.
- Run inference (including real-time use-cases), then review and correct results in Annolid.

## Export strategy
To combine outputs cleanly across tools, pick one “source of truth” format:
- **LabelMe JSON** (best for review/editing in Annolid)
- **CSV** (best for analysis in Python/R)
- **COCO / YOLO** (best for training external models)
