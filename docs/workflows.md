# Workflows

## Standard Video-First Workflow

Annolid supports direct video labeling without mandatory frame extraction.

1. Prepare your video data.
2. Open video in Annolid.
3. Jump to a representative frame.
4. Label instances on that frame.
5. Run tracking/propagation.
6. Review and correct predictions.
7. Export annotations or downstream training data.

## Practical Notes

- Use extraction tools only when you explicitly need frame-level datasets.
- Keep annotations incremental and review high-motion sections early.
- Prefer short review loops over long unattended tracking runs.

## Advanced References

- Book walkthrough: [book/content/README.md](https://github.com/healthonrails/annolid/blob/main/book/content/README.md)
