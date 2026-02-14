# SOC/KYC-Oriented Deepfake Detection Benchmark

## Diffusion-Based Detection Under Operational Constraints

------------------------------------------------------------------------

# 1. Research Overview

## 1.1 Motivation

Modern diffusion models (e.g., Stable Diffusion and proprietary Nano
variants) can generate highly realistic:

-   Government ID documents
-   Selfie-style headshots for KYC onboarding
-   Natural scene images used as supporting evidence

Traditional detection research evaluates models under clean laboratory
conditions and reports AUROC as the primary metric. However, real-world
KYC and SOC pipelines operate under:

-   Strict false-positive budgets
-   Pipeline transformations (e.g., JPEG compression)
-   Evolving unseen generator families
-   Low base-rate attack scenarios

This work proposes a benchmark explicitly designed around SOC/KYC
operational realities.

------------------------------------------------------------------------

# 2. Threat Model

## 2.1 Adversary Capabilities

The adversary can: - Generate ID-like, selfie-like, or scene-like images
using modern diffusion models. - Use unseen diffusion families (Nano
2.5, Nano 3). - Exploit pipeline transformations (JPEG recompression).

The adversary cannot: - Access detector thresholds. - Modify deployment
thresholds post-hoc. - Influence validation calibration.

## 2.2 Deployment Assumptions

-   Detector deployed as standalone control.
-   False positive budget is limited (1% or 0.1%).
-   Calibration performed only on historical known diffusion family
    (Stable Diffusion).
-   Unseen diffusion families are unknown at deployment time.

------------------------------------------------------------------------

# 3. Dataset Structure

## 3.1 Scenarios (3 tasks)

1.  Document / ID Verification
2.  Headshot / Selfie-based KYC
3.  Natural Scene Evidence

## 3.2 Data Counts (Total)

-   Real: 1500
-   Stable Diffusion (Seen): 1500
-   Nano 2.5 (Unseen): 1000
-   Nano 3 (Unseen): 500

Total: 4500 images

All data is balanced across scenarios.

------------------------------------------------------------------------

# 4. Dataset Splitting Protocol

Seed: 825

## 4.1 Seen Generators (Real + Stable Diffusion)

60% Train 20% Validation 20% Test_seen

## 4.2 Unseen Generators (Nano 2.5, Nano 3)

100% Test_unseen

No unseen generator data is used for training or calibration.

------------------------------------------------------------------------

# 5. Detector Architectures

## 5.1 DIRE (Pretrained)

-   Official pretrained ICCV 2023 model
-   No retraining
-   Inference-only mode
-   Preprocessing:
    -   Resize(256)
    -   CenterCrop(224)
    -   ImageNet normalization

Output: Probability of being synthetic

Calibration: Performed only on validation set (Real + SD)

------------------------------------------------------------------------

## 5.2 hfreq (Frequency-Based Detector)

Architecture: - 256×256 RGB input - Convert to luminance - FFT → log
magnitude spectrum - 4-layer shallow CNN - Global pooling + FC

Training: - Train split only - Adam optimizer - BCE loss - Early
stopping using validation AUROC

Calibration: - Validation set only - Same protocol as DIRE

------------------------------------------------------------------------

# 6. Calibration Protocol

Calibration is performed ONLY on validation data (Real + Stable
Diffusion).

Steps: 1. Compute raw detector scores. 2. Compute ROC curve. 3. Select
thresholds for: - FPR = 1% - FPR = 0.1% 4. Freeze thresholds. 5.
Optional logistic regression mapping (score → probability).

Unseen generators are NEVER used in calibration.

------------------------------------------------------------------------

# 7. Evaluation Metrics

Primary metrics (SOC-oriented):

-   TPR @ FPR = 1%
-   TPR @ FPR = 0.1%

Secondary metrics:

-   AUROC
-   Seen vs Unseen degradation
-   Robustness under JPEG compression

------------------------------------------------------------------------

# 8. JPEG Robustness Experiment

## 8.1 Motivation

KYC and SOC pipelines commonly recompress uploaded images (e.g., email,
ticketing systems).

## 8.2 JPEG Variant A (Strict Robustness)

-   Convert all images to JPEG quality=90.
-   No retraining.
-   No recalibration.
-   Apply clean thresholds directly.

This evaluates deployment robustness under pipeline shift.

------------------------------------------------------------------------

# 9. Experimental Matrix

Evaluation is conducted across:

-   Scenario (Doc / Headshot / Scene)
-   Generator family (Seen vs Unseen)
-   Variant (Clean vs JPEG90)
-   Operating point (1% and 0.1% FPR)

------------------------------------------------------------------------

# 10. Expected Outputs

For each detector and variant:

-   scores_clean.csv
-   scores_jpeg90.csv
-   calibration_clean.json
-   metrics_clean.json
-   metrics_jpeg90_variantA.json

Each score CSV contains:

image_id,label,scenario,split,generator_family,score

------------------------------------------------------------------------

# 11. Benchmark Novelty

This benchmark is unique because it:

-   Prioritizes low-FPR operational constraints.
-   Evaluates unseen diffusion family generalization.
-   Introduces pipeline-aware robustness (JPEG90).
-   Emulates real SOC/KYC deployment realities.
-   Avoids calibration leakage.
-   Reports deployment-relevant performance instead of only AUROC.

------------------------------------------------------------------------

# 12. Research Contributions

1.  SOC/KYC-oriented diffusion detection evaluation.
2.  Low-FPR calibration framework.
3.  Unseen diffusion robustness analysis.
4.  JPEG pipeline stress-testing methodology.
5.  Comparative analysis between reconstruction-based and
    frequency-based detectors.

------------------------------------------------------------------------

# 13. Reproducibility Requirements

-   Fixed seed: 825
-   Manifest-driven dataset loading
-   Frozen thresholds after validation
-   Separate clean and JPEG score files
-   No Nano usage in calibration
-   Save per-image scores for full reproducibility

------------------------------------------------------------------------

# 14. Deployment Interpretation

Results should be interpreted in terms of:

-   Alert volume under low base-rate conditions
-   False alarm cost in SOC/KYC
-   Sensitivity degradation under recompression
-   Robustness gap between reconstruction-based and frequency-based
    approaches

------------------------------------------------------------------------

# End of Benchmark Specification
