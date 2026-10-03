# Support Device Removal from Chest X-rays

This project removes medical support devices (tubes, lines, pacemakers) from chest X-ray images using a two-stage pipeline:

1. **Segmentation** — detect and localize devices
2. **Inpainting** — regenerate the masked regions as normal anatomical tissue

## Overview

```
Original CXR (with devices)
        ↓
┌─────────────────────────────┐
│  Segmentation (two models)  │
├─────────────────────────────┤
│  cxr-lines-and-tubes        │ → ETT, NGT, CVC, Swan-Ganz
│  CIED                       │ → Pacemaker, biomonitor
└─────────────────────────────┘
        ↓
   Combined mask
        ↓
┌─────────────────────────────┐
│  Inpainting (Cheff LDM)     │
└─────────────────────────────┘
        ↓
Clean CXR (devices removed)
```

## Pipeline Components

### 1. Segmentation

| Model | Source | Covers |
|---|---|---|
| [cxr-lines-and-tubes](https://huggingface.co/ianpan/cxr-lines-and-tubes) | EfficientNet-B1 + U-Net | ETT, NGT, CVC, Swan-Ganz lines & tips |
| [CIED](https://github.com/Cardio-AI/cied) | ResNet-50 + U-Net (fastai) | Pacemaker / biomonitor body |

Run each:

```bash
# Tubes/lines segmentation
python scripts/cxr_lines_tubes_segmentation.py image.jpg --out outputs/cxr_seg

# Pacemaker segmentation
python scripts/cied_pacemaker_segmentation.py image.jpg --out outputs/cied_seg

# Combine both masks (union)
python scripts/combine_masks.py
```

Output structure per image:

```
outputs/combined_seg/<image_name>/
├── original.png       # original CXR
├── mask.png           # combined binary mask
├── original_256.png   # 256x256 for inpainting
├── mask_256.png       # 256x256 for inpainting
└── result.json        # metadata
```

### 2. Inpainting

Uses **Cheff** latent diffusion model to regenerate masked regions:

```bash
# Unconditional
python scripts/batch_inpainting.py --n-samples 4 --sampling-steps 100

# Text-guided
python scripts/batch_inpainting.py \
    --text-guided \
    --prompt "chest x-ray without any medical devices or tubes" \
    --n-samples 4
```

Output:

```
outputs/inpainted/<image_name>/
├── original.png
├── mask.png
├── inpainted_0.png    # generated samples
├── inpainted_1.png
├── ...
└── comparison.png     # side-by-side grid
```

## Models

| Model | Purpose | Download |
|---|---|---|
| cheff_autoencoder.pt | VAE for latent space | [link](https://syncandshare.lrz.de/getlink/fiQ6wTe7K7otQzyifNh9av/cheff_autoencoder.pt) |
| cheff_diff_uncond.pt | Unconditional LDM | [link](https://syncandshare.lrz.de/getlink/fiE9pKbK38wzEvBrBCk95W/cheff_diff_uncond.pt) |
| cheff_diff_t2i.pt | Text-conditioned LDM | [link](https://syncandshare.lrz.de/getlink/fi4R87B3cEWgSx4Wivyizb/cheff_diff_t2i.pt) |
| cxr-lines-and-tubes | Tubes/lines segmentation | [HF](https://huggingface.co/ianpan/cxr-lines-and-tubes) |
| CIED segmentation.pkl | Pacemaker segmentation | [Zenodo](https://zenodo.org/record/10955502) |

Place Cheff weights in `trained_models/` and CIED weights in `cied/models/`.

## Setup

```bash
conda env create -f environment.yml
conda activate Cheff

# For GPU support (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Citation

If you use this project, please cite the original works:

**Cheff:**
```
@inproceedings{weber2023cascaded,
  title={Cascaded Latent Diffusion Models for High-Resolution Chest X-ray Synthesis},
  author={Weber, Tobias and Ingrisch, Michael and Bischl, Bernd and R{\"u}gamer, David},
  booktitle={PAKDD 2023},
  year={2023}
}
```

**cxr-lines-and-tubes:**
```
@misc{pan2024cxrlines,
  author={Pan, Ian},
  title={CXR Lines and Tubes},
  howpublished={\url{https://huggingface.co/ianpan/cxr-lines-and-tubes}},
  year={2024}
}
```

**CIED:**
```
@article{busch2024cardiac,
  title={Open Access Data and Deep Learning for Cardiac Device Identification},
  author={Busch, Felix and Bressem, Keno K. and others},
  journal={Radiology: Artificial Intelligence},
  year={2024}
}
```

## Acknowledgements

- [Cheff](https://github.com/saiboxx/chexray-diffusion) — latent diffusion backbone
- [cxr-lines-and-tubes](https://github.com/i-pan/cxr-lines-and-tubes) — tubes/lines segmentation
- [CIED](https://github.com/Cardio-AI/cied) — cardiac device segmentation
- [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) — base LDM implementation
