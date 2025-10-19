# LAB-Net Shadow Removal (ISTD+)

A simplified reimplementation of **LAB-Net (Yang et al., 2022)** for single-image shadow removal using the **Adjusted ISTD+ dataset**.

The model works in **LAB color space**:
- **L channel → illumination / shadow correction**
- **a,b channels → color consistency / reflectance**

It’s lightweight, GAN-free

---

## Setup

Project Structure:

labnet-shadow-removal/
  data/ISTD_plus/
    train_C_fixed_ours/
    test_C_fixed_official/
    test A
    test B
    test C
    train A
    train B
    train C
    (All the test A,B,C and train A,B,C contain samples from the original ISTD dataset)
  src/
    dataset.py
    model.py
    losses.py
    train.py
    eval.py
    utils.py
  requirements.txt
  README.md
```

###  Dataset
Download the **Adjusted ISTD+ Dataset** from the [Stony Brook CVLab (SID repo)](https://github.com/cvlab-stonybrook/).
Unzip so that you have:
```
data/ISTD_plus/train_C_fixed_ours/*.png
  (1-1 shadow, 1-2 mask, 1-3 GT)
data/ISTD_plus/test_C_fixed_official/*.png
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # (Linux/Mac)
.venv\Scripts\activate     # (Windows)
pip install -r requirements.txt
```

---

## Training

```bash
python -m src.train --data_root data/ISTD_plus --epochs 6 --bs 8 --lr 2e-4 --resize 256 256
```

Output:
- Checkpoints → `runs/ckpts/`
- Sample grids → `runs/samples/`

---

## Evaluation

```bash
python -m src.eval --ckpt runs/ckpts/labnet_stepXXXX.pt --data_root data/ISTD_plus --bs 4 --resize 256 256
```

Metrics are saved to `runs/eval/metrics.csv` and `runs/eval/summary.txt`, with qualitative grids in `runs/eval/images/`.

---

## Key References

- **Yang et al., 2022.** *“LAB-Net: LAB Color-Space Oriented Lightweight Network for Shadow Removal.”* arXiv:2208.13039.  
- **Wang et al., 2018.** *“Stacked Conditional GANs for Shadow Detection and Removal.”* CVPR.  
- **Adjusted ISTD+ Dataset (SID Repo, Stony Brook CVLab).**

---

##  Components
- `dataset.py` → Loads ISTD+, performs RGB→LAB conversion, mean-match, and masking.
- `model.py` → Shared encoder + dual decoders (L and ab branches).
- `losses.py` → Masked L1, SSIM, and outside-mask consistency.
- `train.py` → Training loop with periodic checkpointing and previews.
- `eval.py` → Computes PSNR/SSIM/RMSE inside shadow mask.
- `utils.py` → Conversions, metrics, and visualization helpers.

---

##  Default Hyperparameters
| Parameter | Value |
|------------|--------|
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| LR | 2e-4 |
| Batch Size | 8 |
| Epochs | 6 |
| λ_ab | 1.0 |
| λ_ssim | 0.2 |
| λ_out | 0.05 |

---


