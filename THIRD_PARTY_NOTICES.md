# Third-party notices

This repository contains or is derived from the following third-party material.

## denoising-diffusion-pytorch (MIT License)

`model/denoising_diffusion_pytorch_modified/` is a modified copy of
[denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch),
Copyright (c) 2020 Phil Wang, distributed under the MIT License. The original license text is
kept in `model/denoising_diffusion_pytorch_modified/LICENSE`. The modifications are listed in
the header of `denoising_diffusion_pytorch.py`. The modified module also imports the unmodified
`denoising_diffusion_pytorch` package, which is installed from PyPI (`requirements.txt`).

## Digitized reference data, Mirocha et al. (2018) (CC BY 4.0)

The reference curves and measurements shown in `docs/figures/validation_meanU_Ug7.15_z0.1.png`
were digitized by us from Fig. 9 of

> J. D. Mirocha et al., *Large-eddy simulation sensitivities to variations of configuration and forcing
> parameters in canonical boundary-layer flows for wind energy applications*,
> Wind Energy Science 3, 589–613 (2018), https://doi.org/10.5194/wes-3-589-2018.

The article is distributed under the Creative Commons Attribution 4.0 License
(https://creativecommons.org/licenses/by/4.0/). The digitized values are approximate
(read from the published figure) and are shown with this attribution.

## Xcompact3d

The files in `les/` are input files and job scripts for
[Xcompact3d](https://github.com/xcompact3d/Incompact3d) (BSD 3-Clause License). No Xcompact3d
source code is included. Xcompact3d must be obtained separately, and users should cite
Bartholomew et al., SoftwareX 12 (2020) 100550, and Laizet & Lamballais, J. Comput. Phys. 228
(2009) 5989–6015.

## Python dependencies

The packages listed in `requirements.txt` (PyTorch, NumPy, SciPy, Matplotlib, einops, etc.) are
not redistributed here. They are installed separately and are subject to their own licenses.
