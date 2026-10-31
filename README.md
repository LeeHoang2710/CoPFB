# CoPFB

Content-Preserving Principal Feature Blending (CoPFB) is a training-free method for style personalization with visual autoregressive models. It transfers appearance from a reference image while protecting the object, geometry, and composition specified by a content prompt.

This repository contains the reproducible Infinity-2B implementation, paper-aligned experiments, evaluation utilities, and the project page.

## Method

CoPFB runs a clean content stream and an editable stream with the same prompt. At selected autoregressive scales it:

1. extracts the clean cumulative content feature;
2. constructs a low-rank PFB style residual from the reference and editable features;
3. estimates a centered low-rank content basis with SVD;
4. removes the residual component aligned with that basis;
5. injects the remaining style signal with a scale-dependent strength; and
6. optionally applies Structural Attention Correction (SAC) during refinement.

The released paper configuration uses rank-one PFB at the first two scales, content-projected rank-one refinement at the next three scales, a 0.75 strength decay, and no injection after that. The implementation also supports fixed or variance-threshold content ranks, arbitrary injection schedules, projection strength, optional mean preservation, and per-scale projection diagnostics.

## Repository layout

```text
.
├── Infinity/                         # vendored FoundationVision Infinity source
├── src/var_soict/                    # reusable CoPFB, SAC, loading, and metrics code
├── scripts/                          # reproducible generation and evaluation CLIs
├── notebooks/                        # Colab/notebook entry points
├── prompts/                          # Parti content prompts and evaluation manifests
├── styles/                           # FineStyle/reference images and metadata
├── csd100/                           # step-wise analysis reference pairs
├── tests/                            # CPU tests for feature projection invariants
├── assets/                           # figures used by the project page
└── index.html                        # GitHub Pages project page
```

## Installation

Use a CUDA-enabled environment compatible with the Infinity dependencies. The model requires a GPU with enough memory for Infinity-2B inference.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r Infinity/requirements.txt
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

FlashAttention is optional. When a compatible wheel is unavailable, the patched Infinity attention code falls back to PyTorch scaled-dot-product attention. The vendored model code also preserves quantized `Linear.forward()` paths so GGUF-loaded weights are dequantized correctly.

## Model weights

Download the official Infinity-2B checkpoint, the compatible d32reg VAE, the optional d64 VAE, and FLAN-T5-XL:

```bash
python3 scripts/download_infinity_weights.py --output-dir weights
```

The released `infinity_2b_reg.pth` checkpoint must be paired with `infinity_vae_d32reg.pth`. The d64 file is retained for experiments but is not shape-compatible with this checkpoint.

## Validate the dataset bundle

The default quantitative bundle contains 190 content prompts and 10 evaluation styles, producing 1,900 prompt/reference pairs:

```bash
python3 scripts/run_content_ortho_batch.py --validate-only
python3 scripts/run_content_projection_batch.py --validate-only
```

The larger prompt preparation bundle also contains 21 style references and 3,990 content/style combinations. CSV paths are resolved relative to the repository root.

## Reproduce the paper configuration

Run one GPU smoke test first:

```bash
python3 scripts/run_content_ortho_multiscale_decay_batch.py \
  --limit 1
```

Then resume the complete 1,900-image run:

```bash
python3 scripts/run_content_ortho_multiscale_decay_batch.py
```

The command is resumable. Existing case folders are skipped unless `--overwrite` is supplied. Each completed case contains `generated.png`, `comparison.png`, and `metadata.json`.

The main single-scale projected-PFB runner is:

```bash
python3 scripts/run_content_projection_batch.py \
  --inject-step 3 \
  --style-rank 1 \
  --content-rank 1 \
  --sac
```

For adaptive content bases, replace the fixed rank with an energy threshold:

```bash
python3 scripts/run_content_projection_batch.py \
  --content-rank 0 \
  --content-variance-threshold 0.9
```

Run the original PFB+SAC baseline with:

```bash
python3 scripts/run_pfb_sac.py \
  --prompt "a train, in oil painting style" \
  --style-image styles/style_05_oil_painting_style.png
```

## Evaluation and diagnostics

Evaluate generated images with CLIP text similarity, reference-image similarity, harmonic score, and content-leakage diagnostics:

```bash
python3 scripts/evaluate_content_ortho_clip.py \
  --generated-root outputs/content_ortho_multiscale_decay_0_1_2_3_4_5 \
  --output-dir outputs/content_ortho_multiscale_decay_0_1_2_3_4_5/metrics
```

The projected feature API can return diagnostics for each sample, including selected content rank, explained variance, removed fraction, spectral gap, and orthogonality residual:

```python
from var_soict.feature_hypotheses import projected_pfb_content_blend

edited, diagnostics = projected_pfb_content_blend(
    generation_feature,
    style_feature,
    content_feature,
    style_rank=1,
    content_rank=1,
    projection_strength=1.0,
    preserve_mean=False,
    return_diagnostics=True,
)
```

## Tests and static checks

The tests run entirely on CPU and verify shape preservation, orthogonality, the zero-projection PFB limit, degenerate one-token behavior, and adaptive-rank selection:

```bash
python3 -m pytest -q tests
python3 -m compileall -q src scripts Infinity/infinity/models
```

Full image generation requires CUDA, model weights, and the Infinity runtime; it is intentionally not run as part of the CPU test suite.

## Paper and project page

- Paper source: `Source/camera.tex` in the companion camera-ready bundle.
- Project page: <https://leehoang2710.github.io/CoPFB/>
- Code: <https://github.com/LeeHoang2710/CoPFB>

## Citation

```bibtex
@inproceedings{bui2026copfb,
  title     = {Content-Preserving Principal Feature Blending for Training-Free Style Personalization},
  author    = {Le-Hoang Bui and Hoang-Nguyen Cao and Trung-Nghia Le and Minh-Triet Tran and Dinh-Khoi Vo},
  booktitle = {Proceedings of SOICT 2026},
  year      = {2026}
}
```
