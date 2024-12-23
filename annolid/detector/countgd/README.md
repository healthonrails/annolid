# CountGD: Multi-Modal Open-World Counting

CountGD is a multi-modal open-world counting tool developed by N. Amini-Naieni, T. Han, and A. Zisserman. This tool is designed to provide accurate counting in various scenarios.

## Links
- [GitHub Repository](https://github.com/niki-amini-naieni/CountGD/tree/main)
- [Hugging Face Space](https://huggingface.co/spaces/nikigoli/countgd)

## Troubleshooting
If you encounter the following error:

[ERROR  ] canvas:_predict_similar_rectangles:1504- Error in CountGD: name '_C' is not defined
```
cd annolid/annolid/detector/countgd/models/GroundingDINO/ops
python setup.py build install
python test.py  # should result in 6 lines of * True
```

## Reference

```bibtex
@InProceedings{AminiNaieni24,
  author = "Amini-Naieni, N. and Han, T. and Zisserman, A.",
  title = "CountGD: Multi-Modal Open-World Counting",
  booktitle = "Advances in Neural Information Processing Systems (NeurIPS)",
  year = "2024",
}
```
