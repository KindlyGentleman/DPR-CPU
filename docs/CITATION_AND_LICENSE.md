# Citation and license

## Citation

When using DPR or weighted partial convolution, please cite:

> S. Y. Lee, D. H. Cho, C. Jung, D. Sung, D. Nam, S. Kim, and C. Song, Deep-learning real-time
> phase retrieval of imperfect diffraction patterns from X-ray free-electron lasers,
> *npj Comput. Mater.* **11**, 68 (2025). <https://doi.org/10.1038/s41524-025-01569-7>

```bibtex
@article{lee2025npjcompumats,
  title   = {Deep-learning real-time phase retrieval of imperfect diffraction patterns from X-ray free-electron lasers},
  author  = {Lee, Sung Yun and Cho, Do Hyung and Jung, Chulho and Sung, Daeho and Nam, Daewoong and Kim, Sangsoo and Song, Changyong},
  journal = {npj Computational Materials},
  volume  = {11},
  pages   = {68},
  year    = {2025},
  doi     = {10.1038/s41524-025-01569-7}
}
```

For the phase retrieval algorithms, please also cite the references of
[PhaseRetrieval](https://github.com/sungyun98/PhaseRetrieval).


Scientific visualization/review workflow guidance: Kassis, Agarwal, He, Patel and Brueckner
(2026), [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065).

## License

This code is released under the BSD 2-Clause License (`LICENSE.txt`), except for the
third-party code below, which keeps its original license (full text in `LICENSES/`); the
changes made are stated at the top of each file.

| File | Source | License |
|---|---|---|
| `deeppr/weightedpartialconv2d.py` | derived from [NVIDIA/partialconv](https://github.com/NVIDIA/partialconv) `models/partialconv2d.py` | BSD 3-Clause, Copyright (c) 2018 NVIDIA Corporation |
| `deeppr/loss.py` (`VGG19Partial`, `gram_matrix`) | adapted from [NVIDIA/partialconv](https://github.com/NVIDIA/partialconv) `models/loss.py` | BSD 3-Clause, Copyright (c) 2019 NVIDIA Corporation |
| `deeppr/ffc.py` (FFC layers) | modified from [pkumivision/FFC](https://github.com/pkumivision/FFC) `model_zoo/ffc.py` | Apache 2.0, Copyright 2022 Lu Chi |
| `deeppr/ffc.py` (`FFCResNetBlock`, `ConcatTupleLayer`) | modified from [advimman/lama](https://github.com/advimman/lama) `saicinpainting/training/modules/ffc.py` | Apache 2.0, Copyright 2021 Samsung Research |
| `deeppr/asam.py` | copied from SamsungLabs/ASAM `asam.py` (repository no longer online; [Software Heritage archive](https://archive.softwareheritage.org/swh:1:rev:f156a680171db16d551c0d85cba2514fa3bff6a2)) | Apache 2.0, Copyright 2021 Samsung Research |

The pretrained VGG19 weights used by the perceptual loss are downloaded by torchvision at run
time and are not distributed with this repository.
