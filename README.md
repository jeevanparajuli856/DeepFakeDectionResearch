# Security Under Diffusion Benchmark

Security-focused diffusion deepfake benchmark for document, headshot, and scene verification. The repo evaluates two detectors under seen and unseen generator shifts, with both PNG and JPEG-90 variants.

## Detectors

- **DIRE**: pretrained diffusion reconstruction error, inference-only
- **HFreq**: frequency-domain CNN trained per scenario

## Quick start

1) Install dependencies from [requirements.txt](requirements.txt).

```
python -m venv venv
venv/Scripts/python.exe -m pip install -r requirements.txt
```

2) Prepare data and manifests.

- PNG masters: [data/images/](data/images/)
- JPEG-90 mirrors: [data/images_jpeg/](data/images_jpeg/)
- Master manifest: [data/manifests/master_manifest.csv](data/manifests/master_manifest.csv)

3) Run the full benchmark.

```
python -m runners.run_all --dire_ckpt detectors/checkpoints/imagenet_adm.pth --plot
```

4) Run JPEG-only evaluation (uses existing models and calibration).

```
python -m runners.run_all --jpeg --dire_ckpt detectors/checkpoints/imagenet_adm.pth --plot --plot_variant jpeg
```

Outputs are written under [outputs/](outputs/) by default.

## Data layout

Expected structure (relative paths are stored in manifests):

```
data/
	images/
		doc/
		headshot/
		scene/
	images_jpeg/
		doc/
		headshot/
		scene/
```

## Manifest schema

Manifests are CSVs with this header:

```
image_id,path,label,scenario,source,generator_family,split
```

Field meanings:

- `label`: 0 = bona fide, 1 = AI-generated
- `scenario`: `doc`, `headshot`, `scene`
- `generator_family`: `real`, `sd`, `nano25`, `nanopro`
- `split`: `train`, `val`, `test_seen`, `test_unseen`

## Dataset splits

- Seen: `real` and `sd`
- Unseen: `nano25` and `nanopro`
- Train and validation use only seen generators

## Checkpoints

Available checkpoints live under [detectors/checkpoints/](detectors/checkpoints/).
Use the one that matches your experiment goal.

## Helper scripts (optional)

Local helper scripts may exist under [scripts/](scripts/) for manifest building, sanity checks, and JPEG conversion. These are optional utilities and may not be tracked in the repo.

## Paper and citation

An EPUB version of the paper is included in this repository for citation. Add the EPUB file to the repo root or a dedicated paper folder and keep the filename stable for citation.

If you use this benchmark or the accompanying paper, please cite or contact:

Jeevan Parajuli  
Department of Computer Science  
University of Louisiana Monroe  
Monroe, LA, USA  
parajulij@warhawks.ulm.edu

## Recommended citation text

Jeevan Parajuli. Security Under Diffusion Benchmark. 2026. EPUB available in the project repository.