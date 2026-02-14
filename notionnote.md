# DeepFake Research — Full Notion Export (All Subpages)

> Workspace tree exported:
> - DeepFake Research
> 	- Plan For General Comparison using Nano
> 	- DeepFake Security

---

## 0) Abstracts (two variants)

### 0.1 Security Under Diffusion (KYC/SOC framing)
**Security Under Diffusion: Evaluating Pretrained DIRE and Frequency‑Based Deepfake Detectors for KYC and SOC Operations**

This work proposes an empirical security‑focused study of deepfake image detection in realistic KYC and SOC workflows. Modern diffusion models such as Stable Diffusion and proprietary Nano variants can synthesize highly convincing ID‑like documents, headshots, and contextual scene images, raising the question of whether current detectors remain reliable under these evolving threats. The study evaluates a pretrained diffusion‑based detector (DIRE‑style) and a lightweight frequency‑domain model across three scenarios that are directly relevant for identity verification and fraud:

1. bona fide ID and travel documents from a privacy‑aware ID dataset versus diffusion‑generated ID‑style images,
2. bona fide human profile photos versus diffusion‑generated headshots mimicking selfie‑based KYC,
3. bona fide natural scenes versus diffusion‑generated scenes representing supporting evidence images.

All images are standardized to a low‑resource‑friendly resolution, and the evaluation focuses on AUROC and true positive rates at low false positive rates to reflect operational constraints. By analyzing performance on both seen (Stable Diffusion) and unseen diffusion families (Nano 2.5, Nano Pro) under consistent preprocessing and calibration, the work aims to quantify how much detection accuracy degrades as generators evolve, and to highlight where existing DIRE and frequency‑based approaches are insufficient as standalone controls in KYC/SOC pipelines.

### 0.2 Beyond Stable Diffusion (Nano Banana framing)
**Beyond Stable Diffusion: Generalization of DIRE‑Style and Frequency‑Based Detectors to Nano Banana 2.5 and Pro**

This paper presents a focused empirical study on how existing AI‑generated image detectors generalize to recent foundation‑model diffusion generators, specifically Google’s Nano Banana 2.5 (Gemini 2.5 Flash Image) and Nano Banana Pro (Gemini 3 Pro Image). We construct a balanced photographic dataset of real images, Stable Diffusion–generated images, Nano Banana 2.5 images from a public corpus, and a smaller but diverse set of Nano Banana Pro images generated via API, all using a shared prompt family covering people, indoor/outdoor scenes, and everyday objects.

On this dataset, we evaluate four representative detector families:
- a DIRE‑style detector using pretrained diffusion reconstruction features,
- a supervised frequency‑domain CNN,
- a lightweight autoencoder reconstruction‑error baseline,
- a simple training‑free high‑frequency energy score.

Zero‑shot experiments train detectors only on real and Stable Diffusion images and measure performance on Nano Banana 2.5 and Nano Banana Pro. Few‑shot adaptation with a small subset of Nano Banana 2.5 partially closes gaps but does not fully recover performance on Nano Banana Pro. The results suggest current detectors are not yet reliable for modern Gemini‑based generators and underline the need for updated benchmarks.

---

## 1) “DeepFake Security” — scenario-first experimental design

### 1.1 Experiment (scenario design and publishable framing)

You can support three scenarios with ~1,500 images each by mixing realistic synthetic/stock “real” data plus your SD/Nano generations, then framing the work as scenario‑driven robustness evaluation of DIRE and frequency detectors.

#### Scenario design (3 × 1,500 images)
Aim for ~750 real‑style + 750 generated per scenario.

##### Scenario 1 – ID / travel documents
Focus: KYC and border‑control fraud.
- Real‑style images
	- Use a synthetic ID/Travel document dataset like SIDTD (IDs/passports, multiple layouts, legit vs forged).
	- Sample ~750 bona fide document images (treat as “real” for binary real vs AI‑generated task).
- Generated images
	- Generate ~750 ID‑style images with SD and Nano (portrait + text blocks + logos) to resemble SIDTD style.
- Why it works
	- Synthetic ID sets exist because real IDs are sensitive and are accepted in doc‑verification research.

##### Scenario 2 – Badges / profile & meeting
Focus: enterprise access badges and remote identity.
- Real‑style images
	- Use a public profile/portrait dataset (e.g., Human Profile Photos on Kaggle) and crop or layout them into badge-like or meeting frames.
	- Build ~750 real‑style images.
- Generated images
	- Use SD/Nano to create ~750 headshot and “video-call style” portraits.

##### Scenario 3 – General scene / context images
Focus: wider misinformation (scenes, objects, news-like pictures).
- Real‑style images
	- Sample ~750 natural images (COCO-like or other open natural corpora).
- Generated images
	- Use SD/Nano to generate ~750 general images with prompts similar to real subset.
- Why include this
	- Modern “deepfakes” include not just faces, but fabricated scenes/events.

Total: ~4,500 images (1,500 per scenario).

#### Integrating DIRE and frequency detectors
Use the same detector pair across all three scenarios.
- DIRE (or DistilDIRE):
	- Use public code/checkpoints as frozen detectors, get a fake score.
	- Calibrate thresholds separately per scenario.
- Frequency-based model:
	- Train a small frequency model per scenario or unified with scenario reporting.

#### Splits and metrics (example)
For each scenario (1,500 images):
- Train (frequency model): ~500
- Validation (freq + DIRE threshold): ~200–250
- Test: ~750–800 (put most Nano Pro here for scarce unseen generator)
Metrics:
- AUROC
- TPR at 1% / 0.1% FPR
- Report per scenario and per generator family (SD vs Nano 2.5 vs Nano Pro)

#### Why publishable
Strengths:
- Three clear security scenarios
- Modern diffusion models
- Consistent comparison: frozen DIRE vs small frequency model
Requirements:
- Be explicit about synthetic/stock nature of “real” images
- Limit claims to patterns under these scenarios/generators

---

### 1.2 Synthetic ID (reviewer-safe labeling and justification)

Reviewers will accept synthetic ID/document images as “real/bona fide” if labels are defined clearly and aligned with ID-forensics practice.

#### Define “real” vs “deepfake” IDs
- Many ID-forensics papers use synthetic/fantasy IDs as bona fide due to privacy.
- For this work (binary labels):
	- Bona fide document image = unmanipulated synthetic/fantasy ID or travel doc from established dataset.
	- AI-generated document image = SD/Nano diffusion image intended to spoof an ID.

Clarify: “bona fide” means legitimate sample by the issuing process proxy, not “non-synthetic pixels.”

#### Avoid confusion
- Separate pipelines:
	- Bona fide IDs: SIDTD-like datasets (legitimate doc proxies).
	- Deepfake IDs: generated by diffusion models via prompts.
- Clear wording:
	- Use “ID-style deepfake” / “diffusion-generated ID.”
	- Reserve “bona fide” for SIDTD-style corpora as those datasets do.

#### Methodology justification (preempt reviewers)
- Cite privacy constraints and existing synthetic ID practice.
- Threat model:
	- Legitimate users upload camera-captured IDs.
	- Attackers may upload fully AI-generated IDs.
- Limit claim:
	- Not studying all fraud types, only diffusion-based synthetic IDs.

---

### 1.3 DataSet Prepare (step-by-step operational plan)

#### Step 1 — Class + scenario design
- Binary label:
	- 0 = bona fide (legitimate)
	- 1 = AI-generated (SD / Nano 2.5 / Nano Pro)
- Scenarios:
	1) ID/docs
	2) badges/profile/meeting faces
	3) general scenes/news-style

Each scenario ~1,500 images: ~750 bona fide + ~750 AI-generated.

#### Step 2 — Collect bona fide images
Scenario 1:
- Choose SIDTD or similar.
- Sample ~750 bona fide docs.
- Store: `data/id_docs/bona_fide/`

Scenario 2:
- Choose portrait datasets.
- Prep ~750 images (crop; optionally overlay badge template).
- Store: `data/badge_meeting/bona_fide/`

Scenario 3:
- Choose open natural image dataset subset.
- Sample ~750.
- Store: `data/scenes/bona_fide/`

#### Step 3 — Generate SD/Nano images (mirror style)
General settings:
- Fix size and format.
- Per scenario:
	- SD: ~350–400
	- Nano 2.5: ~350–400
	- Nano Pro: ~150–200 (mostly test)

Folders:
- `.../generated/sd/`
- `.../generated/nano25/`
- `.../generated/nanopro/`

#### Step 4 — Clean/normalize/label
- Remove broken generations.
- Normalize:
	- same resolution
	- RGB
	- consistent format
- Create CSV/JSON index:
	- `file_path, scenario, label, generator(real/sd/nano25/nanopro)`

#### Step 5 — Train/val/test splits (per scenario)
- Train: ~500 (for frequency model)
- Val: ~200–250 (threshold calibration)
- Test: ~750–800 (most Nano Pro here)
Ensure balance and put most Nano Pro in test.

#### Step 6 — Augmentations (optional)
- JPEG compression 70–90
- slight resize/crop
- mild noise/blur
Either train-only augmentations or robustness evaluation copies.

#### Step 7 — Document for paper
- Scenario description and security meaning
- Data sources and “bona fide” alignment
- Counts per scenario + SD/Nano distribution
- Privacy note

---

### 1.4 Prompt for SD and Nano (templates)

#### SD (runwayml/stable-diffusion-v1-5) — ID/doc prompts
- Photorealistic travel document for a \<age>-year-old \<gender> with \<skin tone> skin, lying flat on a clean table with \<key objects>, front side fully visible, neutral studio light, sharp focus, realistic colors.
- National ID card of a \<age>-year-old \<gender>, top-down view on a neutral desk, card centered, even lighting, crisp text and edges, high-detail realistic photo.

#### Nano 2.5 — ID/doc prompts
- Realistic photo of a passport for a \<age>-year-old \<gender>, card lying flat on a plain surface with \<key objects>, front side visible, soft neutral light, clear details.
- ID card for a \<age>-year-old \<gender> on a light table, top view, card centered, natural colors, readable text, realistic style.

#### Nano 3 (Nano Pro) — ID/doc prompts
- Ultra realistic national ID card for a \<age>-year-old \<gender> with \<skin tone> skin, flat on a clean desk with \<key objects>, front side up, neutral lighting, fine print and security details visible.
- Photorealistic passport for a \<age>-year-old \<gender>, lying open on a smooth surface, top-down shot, sharp edges, legible text, subtle shadows, true-to-life colors.

#### Headshot prompts (SD / Nano2.5 / Nano Pro)
SD:
- Professional headshot ... LinkedIn style ... DSLR 85mm lens look ... shallow DoF ... soft bokeh ... ultra realistic photo.
- Studio portrait ... softbox lighting ... visible skin texture ...

Nano 2.5:
- Professional business headshot ... soft blurred office background ...
- Upper-body portrait ... natural daylight ...

Nano Pro:
- Hyper realistic studio headshot ... catchlights ... detailed skin texture ...
- Photorealistic portrait ... rim light ... magazine-cover quality ...

#### Scene prompts (SD / Nano2.5 / Nano Pro)
SD:
- Street photo ... cars, pedestrians ... lighting/time ... photorealistic
- Interior photo of modern office ... wide-angle ... realistic detail

Nano 2.5:
- Park ... daylight ... natural shadows ...
- Cozy living room ... warm lamp light ...

Nano Pro:
- Ultra realistic evening city street ... wet pavement ... reflections ... cinematic
- Photorealistic office interior ... daylight ...

#### Attribute variance
Keep same sampling across models:
- \<age> in {20,30,40,50,60,70}
- \<gender> in a fixed list defined in paper
- \<skin tone> descriptors: light/medium/deep
- \<hair description>, \<clothing style>
- \<lighting/time of day>, \<scene type>, \<key objects> from fixed enumerations

---

### 1.5 DataSetFailure SD (threat-model limitation write-up)

Phrase as a threat-model limitation, not weakness.

- SIDTD templates emulate real KYC docs.
- Off-the-shelf SD cannot exactly reproduce templates/security features; SD “ID deepfakes” differ in layout and typography.
- Interpret SD/Nano ID deepfakes as plausible ID-like images, not pixel-perfect clones.
- This matches realistic attackers: convincing-looking IDs without exact templates.

Conclusions framing:
- Study evaluates separating bona fide SIDTD IDs from diffusion-generated ID-like images under realistic constraints.
- As generative models improve, gap may shrink, so results are conservative vs future threats.

Note:
- Headshot generation is “pretty much perfect” relative to document template matching.

---

### 1.6 Expectation (expected AUROC drop on Nano)

Expect AUROC to decrease when testing on Nano 2.5 / Nano Pro compared to tuned generators.

Why:
- Unseen diffusion models hurt generalization.
- DIRE strong but not perfect across dataset + generator shifts.
- Frequency detectors sensitive to generator-specific spectral artifacts; newer diffusion may be cleaner.

How to use in paper:
- Calibrate on SIDTD + SD.
- Report AUROC and Pd@1%FAR (TPR@low FPR) for SD vs Nano 2.5 vs Nano Pro.
- Emphasize performance drops as evidence detectors struggle with newer diffusion families.

---

### 1.7 DeepFake Security — Abstract (repeated here for completeness)
(See section 0.1.)

---

### 1.8 DeepFake Security — Documentation (SynthID positioning)
Key notes:
- Generated Nano images with gemini-2.5-flash-image and gemini-3-pro-image-preview.
- Dataset includes JSON metadata for publication.

SynthID positioning:
- Clarify SynthID is invisible watermark decoded by dedicated services.
- Your detectors do not read SynthID.
- Passive detection still matters for SOC/KYC where watermark decoders may be unavailable and attackers can use non-watermarked generators.
- Present watermarking and passive detection as complementary.

---

### 1.9 DeepFake Security — DataSet File (sources and write-up blocks)

Links:
- Portrait: Human Profile Photos Dataset (Kaggle)
- ID/Document bona fide: SIDTD (Dataverse)
- Scene: CIFAKE dataset (Kaggle)

Write-up blocks included:
- SIDTD justification and sampling (~750 template images)
- Human Profile Photos licensing and use (~750)
- CIFAKE real subset sampling (~750)

Also includes a Stable Diffusion synthetic dataset description:
- 1,500 SD images at 512×512 from `runwayml/stable-diffusion-v1-5`
- 3 scenarios (ID/headshot/scene) 500 each
- Prompt templating and fixed seed 825
- Low VRAM settings and optional downsample to 256×256

---

### 1.10 DeepFake Security — DataSet Framing (preprocessing + PNG vs JPEG plan)

Preprocessing checks:
- Valid loadable files, convert grayscale to RGB.
- Center crop/pad to square (avoid stretching).
- Resize to 256×256 with fixed interpolation.
- Save consistently (PNG master preferred).

Labeling:
- Balanced per scenario
- Record in CSV:
	- file_path, scenario, label (0 real / 1 generated), source (sidtd/profile/cifake/sd/nano25/nanopro)

Reproducibility:
- Fixed seed
- Save selected filename lists in CSV

Filename convention:
`<scenario>_<class>_<source>_<id>.jpg`
Examples given for each scenario/source.

PNG master + JPEG-90 robustness:
- Maintain PNG master set
- Create JPEG q=90 copy for email/upload simulation
- Evaluate performance drops on JPEG as robustness result

---

### 1.11 DeepFake Security — DataSet Final Plan (counts, split, detector writeups, metadata columns)

Counts (as written):
- Real 1500 (ID 500, Photo 500, Scene 500)
- SD 1500 (ID 500, Photo 500, Scene 500)
- Nano 2.5 1000 (ID 334, Photo 334, Scene 333)
- Nano 3 500 (ID 167, Photo 166, Scene 166)

Name schema and source sets:
- scenario ∈ {id, headshot, scene}
- class ∈ {real, fake}
- source:
	- real: {sidtd, profile, cifake}
	- fake: {sd, nano25, nanopro}

Seed: 825

Split plan:
- Single global split
- Shuffle with seed 825
- Stratify by scenario/class/generator
- Save assignments to CSV

Preprocessing:
- 256×256
- Condition A: PNG
- Condition B: JPEG q=90

DIRE-style experiment (spatial/learned):
- Standard CNN/ViT baseline
- Train on Train, tune on Val
- Evaluate on PNG and JPEG
- Report metrics per scenario/source

Frequency-based experiment:
- FFT/DCT features, radial spectrum, HF/LF ratios, HP filter stats
- Simple classifier (logreg/SVM/MLP)
- Evaluate PNG vs JPEG q=90

Metadata.csv recommended columns (filename, scenario, class, source, generator_family, seed, split, format_regular, format_email, path_regular, path_email)

---

### 1.12 DeepFake Security — CSV Format (final schema + examples)

Recommended global CSV columns: