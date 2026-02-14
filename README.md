# SOC/KYC Deepfake Benchmark

This repo runs a SOC/KYC-oriented deepfake benchmark for two detectors:

- DIRE (pretrained, inference-only)
- hfreq (frequency CNN, trained per scenario)

## Dataset expectations

Manifests are CSVs with the schema below:

```
image_id,path,label,scenario,source,generator_family,split
```

Key fields:

- `label`: 0 = bona fide, 1 = AI-generated
- `scenario`: `doc`, `headshot`, `scene`
- `generator_family`: `real`, `sd`, `nano25`, `nanopro`
- `split`: `train`, `val`, `test_seen`, `test_unseen`

PNG masters should live under `data/images/...` and JPEG-90 mirrors under
`data/images_jpeg/...` with identical relative structure.

## Prepare manifests

1) Update `data/manifests/master_manifest.csv` to match your sources.
2) Split using:

```
python scripts/split_dataset.py
```

## JPEG-90 mirror

```
python scripts/make_jpeg90.py
```

## Run full benchmark

```
python -m runners.run_all --dire_ckpt detectors/checkpoints/lsun_stylegan.pth
```

JPEG-only evaluation (uses existing models + calibration):

```
python -m runners.run_all --jpeg --dire_ckpt detectors/checkpoints/lsun_stylegan.pth
```

Generate plots (PNG or JPEG summaries):

```
python -m runners.run_all --dire_ckpt detectors/checkpoints/lsun_stylegan.pth --plot
python -m runners.run_all --jpeg --dire_ckpt detectors/checkpoints/lsun_stylegan.pth --plot --plot_variant jpeg
```

Outputs are written under `outputs/hfreq/<scenario>/` and `outputs/dire/<scenario>/`.