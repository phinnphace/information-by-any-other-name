# RBM Exploratory Protocol: Learning the Shape of the Data

Oct 17, 2026 · @phinn markson

## Purpose and stance

This phase learns the shape of the CASIA data from the data, before any task, label or architecture tells it what to look for. It is an exploratory data analysis in the literal sense, run with a generative model instead of summary plots.

Every model in Studies 1 and 2 had an objective that defined what mattered: a binary label, a six-way label, or a contrastive definition of similar. Each objective is a lens, and the results described the data as seen through that lens. A restricted Boltzmann machine (RBM) models p(v), the distribution of the images themselves. It has no labels and no imposed notion of similarity, so whatever structure it finds came from the images.

**How this departs from standard practice:**

- No accuracy, no benchmark, no train/test performance claim. Labels are used only after training, to stratify descriptive outputs.
- Questions, not hypotheses. Nothing here is tested for significance; the outputs are looked at, described and counted (explored).
- No probes. The model's own decoding path is the view onto its hidden code (Background: Looking instead of probing).
- The instrument is validated before it is trusted (Section: Instrument validation), as the linear probe was.
- Every transformation between the handwritten page and the model's input is logged with its known or unknown distance from the observation (Section: Data dictionary, part 1).

**What this phase will not claim:** that one pretraining regime, unit type or data source is better. Its output is a description of the data's structure and of what each hidden-unit type keeps and discards. That description sets the design of the next phase.

## Background

Studies 1 and 2 left a contradiction that no task-driven model can resolve: isolated pretraining always produced sparser conv3 representations, yet the 'better' model flipped between characters.

| Measure | Study 1 (打) | Study 2 (的) |
| --- | --- | --- |
| Isolated model (A) | 61.2% | 90.4% (relu3 probe) |
| Contextual model (B) | 91.4% | 66.3% (relu3 probe) |
| Better model | B | A |
| conv3 fraction negative, A | 0.844 | 0.992 |
| conv3 fraction negative, B | 0.780 | 0.854 |
| Top-20 channel overlap, A vs B | 0 of 20 | 0 of 20 |

Study 2 also showed the contextual model's character signal was present before the gate: a pre-ReLU spatial probe reached 80.8% where the post-ReLU pooled probe reached 66.3%. Across 1,000 matched seeds the A − B gap at relu3 had a 95% interval of −0.092 to 0.379, with a sign flip on 68 seeds. Sparsity is a correlate of the pretraining source, not a declarative mechanistic metric for model win/lose.

**Why an RBM, and why this one.** Nair & Hinton (2010) introduced rectified hidden units into RBMs and made the claim later cited as "ReLUs preserve information." Read closely, that claim holds only for zero-bias units, only relative to binary units, and only for units above zero. The paper also gives the diagnostics this phase reuses: learned filters shown as images, and a histogram of how often each hidden unit is active (their Figure 5), which differed between NORB and LFW with the same unit type.

An RBM reconstructs its input. That makes the distance between representation and observation directly visible as a per-pixel residual, which no classifier provides. Hinton's practical guide (2010) is the operating manual for training, and it warns that reconstruction error is a misleading measure of fit. That warning is treated here as an instrument-validation requirement, not a footnote.

**Looking instead of probing.** A linear probe adds an instrument. It sees only what a logistic regression can separate, at one layer, after whatever reduction was chosen first (pooling, Fourier terms, projection), and each of those choices is one more thing to rule out. What it reports belongs to the representation and the probe together. A reconstruction adds no instrument. The hidden-to-visible pass uses the same weights as the visible-to-hidden pass, run the other way, so asking the model to rebuild an image is asking it to complete its own circuit. The output is an image, and the measurement is looking at it. Looking is still an instrument, with its own frame (one observer, the rendering chosen), but that frame is stated, and the picture can be put in front of anyone.

**The path back.** Huffman coding gives each symbol a code whose decoding path is built into the code itself; no separate decoder is learned. The RBM has the same structural property: one weight matrix, W going up and its transpose coming down. Where the two part ways is useful here. Huffman decoding returns the original exactly. The RBM's path back returns p(v|h), a distribution over images, so what comes back differs from what went in, and that difference, shown as a residual, is the object of study. Huffman also gives frequent symbols short codes and rare ones long codes, yet returns every symbol exactly; a lossy model spends its capacity on what is frequent, and rare structure may come back worse or not at all. Han, Mao & Dally (2016), who Huffman-coded a trained network's stored weights, informed this framing; here the idea is carried from the stored model to the representation itself.

**Using the rectified unit where it started.** Nair & Hinton named the rectified linear unit and introduced it to keep more information about the input in an RBM's hidden code: magnitude, not only on or off. Deep feed-forward networks later took it up as a cheap nonlinearity for discrimination (Glorot, Bordes & Bengio, 2011), outside the generative setting it was introduced in. Rectification itself appears earlier (for example Jarrett et al., 2009). This phase uses the unit for the purpose Nair & Hinton gave it.

## Exploratory questions

Seven questions frame what to look at. None carries a predicted answer, and none is tested for significance.

1. **Q1 Structure.** What do the hidden units learn from each data source: stroke fragments, radicals, whole characters, the neighbouring character, or the older writing underneath?
2. **Q2 Activity profile.** How often is each hidden unit active, and does that profile differ by data source, unit type and character?
3. **Q3 Where reconstruction fails.** Which pixels and regions does the model consistently fail to hold, per character and per source?
4. **Q4 Same character, two settings.** Does 打 (or 的) in isolation land near 打 in a bigram in hidden-code space, or do the two occupy separate regions?
5. **Q5 Cross-source reconstruction.** How well does an RBM trained on one source reconstruct images from the other? This is the transfer question restated in generative form.
6. **Q6 What the gate discards.** For rectified units, what do the hidden pre-activations below zero carry, separated into sign pattern, positive magnitude and negative magnitude?
7. **Q7 Rebuilding.** What can each model rebuild, from its own source and from the other, and does character identity remain visible in what it rebuilds?

## Data sources and conditions

One RBM is trained per data condition, on all classes pooled, because the model is unsupervised. Labels are attached afterwards for stratified description only.

| Condition | Study | Source | Content | Images |
| --- | --- | --- | --- | --- |
| S1-ISO | 1 | CASIA-HWDB1.1 training GNT files | 打 plus 扔 扛 扫 托 扣 | 489 (239 打, 50 each distractor) |
| S1-CTX | 1 | CASIA-HWDB2.1 training DGRL files | 打 bigram crops plus negative bigrams, hand-cropped from line images | Counted after extraction |
| S2-ISO | 2 | CASIA-HWDB1.1 training GNT files | 的 plus 地 得 也 他 她 | 1,437 (≈240 per class) |
| S2-CTX | 2 | CASIA-HWDB2.1 training DGRL files | 的 bigram crops plus negative bigrams, hand-cropped from line images | Counted after extraction |

Every condition is drawn from CASIA's own binary files, which store one 8-bit grey channel at original resolution. The contextual crops used in Studies 1 and 2 are not reused. They came from a third-party copy (Teklia, on Hugging Face) that resized every line to 128 pixels high and was stored in colour, and Study 2's were cut by character-width proportion. Changing the source removes those transformations instead of logging them. The Study 1 and 2 crops are kept as an archive, for reference only.

Study 1 is run first, because the original hypothesis held there. Study 2 follows with the same pipeline.

**Provenance.**

| Item | Detail |
| --- | --- |
| Original source | CASIA-HWDB, National Laboratory of Pattern Recognition, Institute of Automation, Chinese Academy of Sciences (Liu et al., 2011) |
| Isolated release | HWDB1.1 training set: 240 GNT files, one file per writer |
| Contextual release | HWDB2.1 training set: DGRL files, one file per page, named writer–page. HWDB2.1 is the text counterpart of HWDB1.1: the same release batch of 300 writers, per CASIA's documentation, so isolated and contextual images share one writer pool |
| Contextual copy obtained from | Hugging Face `OrkaZeta/HWDB2`, file `HWDB2.1Train.zip` (554 MB). An unofficial re-upload with no dataset card. `HWDB2.0Train.zip` (727 MB) was downloaded first on 19 October 2026 and used only to pilot the parser; it is not used in any condition |
| Decoding | `dgrl_extract.py`: one PNG per text line, no resizing or colour change; stops on any layout mismatch. Parser piloted on 3 HWDB2.0 pages (27 lines, checked by eye against their labels). Full HWDB2.1 training set: 1,200 of 1,200 pages decoded, 13,768 lines, 344,048 characters, 0 undecodable labels |
| Terms of use | CASIA releases the databases for academic research; terms to be confirmed on CASIA's official page |
| Considered and not used | `Teklia/CASIA-HWDB2-line` (resized, stored in colour); `UndefinedCpp/casia-char-1` (no writer field, HWDB versions mixed) |

Open items to confirm against the files before running:

- [ ] Writer pairing settled: HWDB1.1 and HWDB2.1 share CASIA's 300-writer release batch, per CASIA's documentation; HWDB2.0 belongs to a different batch and was dropped. Still to check: that GNT file names and DGRL page names use the same writer numbering, so `writer_id` matches across the two sources, and how many of the shared writers wrote 打 or 的 in their text pages
- [ ] Check the zip against any checksums CASIA publishes, since the copy is unofficial
- [ ] Count 打 and 的 occurrences in `lines.csv` to set the S1-CTX and S2-CTX sizes
- [ ] Check whether DGRL line images have a uniform background value or keep paper texture
- [ ] Define ink pixels for the ink and background summaries; the definition depends on whether DGRL backgrounds are uniform (item above)

## Data dictionary, part 1: the proxy ledger

The image a model sees is several transformations away from the handwriting. Each row below is one transformation, with what it changes and whether that distance is known.

The ledger exists because an unmeasured distance cannot be bounded, and one unbounded term makes the total unstateable:

```latex
E_{total} \approx \sum_i d_i \cdot A(k_i, n_i)
```

Here d is the distance a transformation introduces, k whether that distance is known, and n how many later steps depend on it.

| Step | Transformation | What it changes | Delta status | How it gets measured |
| --- | --- | --- | --- | --- |
| 1 | Written with an Anoto pen on paper; page scanned by CASIA | Resolution, intensity quantization | Unknown | Confirm acquisition settings in Liu et al. (2011) |
| 2 | Character segmented by CASIA (HWDB1.1) | Crop boundary, background set to white | Known, unmeasured | Record each GNT image's original width and height |
| 3 | Page segmented into text lines by CASIA (HWDB2.1) | Line boundary; strokes from adjacent lines can intrude | Unknown | Visual coding of intrusion per crop (Part 2, `intrusion`) |
| 4 | GNT and DGRL files decoded to PNG by the project's parsers | Nothing, if the parsers are correct | Lossless | Parser checks: every byte accounted for, every line box inside its page; labels compared with images by eye |
| 5 | Bigram hand-cropped from the line image | Per-image human judgement on crop edges | Known, unmeasured | Crop box logged against the line image (Part 2, `crop_box`), so every crop can be reproduced |
| 6 | Resized to 100 × 100, Lanczos filter | Aspect ratio forced to square; interpolation | Measured in this phase | Original aspect ratio per image vs reconstruction residual |
| 7 | Intensities divided by 255 | Scale only. Both sources are already one grey channel, so there is no colour conversion | Lossless | None needed |
| 8 | Downsampled 100 × 100 to 32 × 32 for the RBM | Fine stroke detail | Measured in this phase | Down-then-up residual per image, before any training |
| 9 | Polarity set (ink dark or ink bright) | Which pixels are near zero; interacts with zero-bias rectified units | Measured in this phase | Run both polarities in the pilot |
| 10 | Per-pixel median removed, divided by one global median absolute deviation (MAD) | Where zero sits for every hidden unit's input | Measured in this phase | Compare against mean/SD scaling in the pilot |

Removed by changing the source, rather than logged: Teklia's resize to 128 pixels high, colour storage, conversion to grey at load time, and Study 2's proportional crop. Each was a transformation of unmeasured distance in Studies 1 and 2.

## Data dictionary, part 2: record variables

The dictionary records provenance and transformations, not presumed features. What the images contain is the output of this phase, so no feature is defined in advance.

**Input records, one row per image:**

| Variable | Type | Values | Meaning |
| --- | --- | --- | --- |
| `image_id` | string | unique | Stable identifier across all runs |
| `condition` | category | S1-ISO, S1-CTX, S2-ISO, S2-CTX | Data condition |
| `class_label` | category | the six characters per study | Used after training only |
| `is_target` | boolean | true, false | 打 or 的 present |
| `release` | category | HWDB1.1 train, HWDB2.1 train | CASIA release the image came from |
| `writer_id` | string | GNT file name; first part of DGRL page name (`001` in `001-P16`) | Writer, for both sources. All four conditions draw from the HWDB1.1/2.1 writer batch, so ISO and CTX are writer-matched within each study |
| `source_file` | string | path | GNT or DGRL file |
| `page_stem` | string | e.g. `001-P16` | Contextual only: writer and page |
| `dgrl_line_index` | integer | 0 and up | Contextual only: which line on the page |
| `line_text` | string | CASIA's label for the whole line | Contextual only: the sentence the crop came from |
| `target_index` | integer | 0 and up | Position of the target character in `line_text`; the same characters at a different position sit in a different sentence |
| `neighbour_chars` | string | characters | The other characters inside the crop |
| `crop_box` | 4 integers | left, top, width, height | Contextual only: the hand-cropped box in line-image pixels |
| `orig_width`, `orig_height` | integer | pixels | Size before resizing (ledger step 6) |
| `intrusion` | category | none, adjacent line, older writing, both | Coded by visual inspection |
| `downsample_residual` | float | 0 and up | Median absolute down-then-up error over ink pixels, with the ink-pixel count (ledger step 8) |
| `fold` | integer | 1 to 5 | Reconstruction hold-out fold |

**Output records, one row per hidden unit per trained model:**

| Variable | Type | Meaning |
| --- | --- | --- |
| `model_id` | string | condition, unit type, hidden size, seed, fold |
| `unit_id` | integer | Hidden unit index |
| `weight_l2` | float | Norm of the unit's weight vector |
| `activity_freq` | float | Fraction of images on which the unit is above zero |
| `activity_freq_by_class` | float per class | Same, stratified by `class_label` |
| `preact_median`, `preact_iqr` | float | Location and spread of the unit's input |
| `frac_negative` | float | Fraction of pre-activations below zero |
| `seed_match_cos` | float | Best cosine match to a unit in another seed's model |

Per-image outputs (reconstruction, residual, hidden code, free energy) are stored keyed on `image_id` and `model_id`. Each reconstruction carries `recon_type` (noise-off, median of draws, or sampled), `n_draws` for median of draws, and, for O8, `nn_train_class`: the class of its nearest training image in pixel space.

Training log, one row per epoch per model: `recon_err_ink` and `recon_err_bg` (per image, the median absolute reconstruction error over ink pixels and over background pixels; logged as the median and interquartile range across images), `grad_norm` (median across mini-batches), `grad_clipped` (count of clipped mini-batches), `active_frac` (fraction of hidden units above zero; called sparsity in the reference script), `free_energy_train` and `free_energy_heldout` (medians). Reconstruction images are saved every 10 epochs. Stage 0 attempts are logged the same way, each with a note of what was changed and why. No means are logged.

## Preprocessing

Preprocessing is kept to the minimum the RBM needs, and every step that is a choice is run both ways in the pilot rather than fixed by convention.

1. **Inventory before transforming.** Record `orig_width`, `orig_height`, `release` and `writer_id` for every image, and `page_stem`, `dgrl_line_index`, `target_index` and `crop_box` for every contextual crop. Code `intrusion` by eye for every contextual crop.
2. **Resolution.** Primary resolution is 32 × 32 (1,024 visible units), following Nair & Hinton. Before training, compute each image's down-then-up residual at 32 and at 48 pixels and report its distribution per condition and class. If fine strokes that separate characters are lost at 32, the 48-pixel run becomes primary.
3. **Polarity.** Run ink-dark and ink-bright in the pilot. Polarity is not lossy, but it decides which pixels sit near zero, which matters to rectified hidden units. Both sources are single-channel grey, so there is no colour arm.
4. **Scaling.** Primary: subtract the per-pixel median and divide by one global median absolute deviation. Comparison: per-pixel mean and global standard deviation, the scaling in Hinton's guide (2010), run so the effect of the choice is seen rather than assumed. Background is most of every image, so a MAD taken over all pixels can come out as zero; the pilot checks this first, and if it does, the scale is taken over ink pixels. Nair & Hinton's wording leaves per-pixel versus global centring ambiguous, so the choice is stated here.
5. **Fit on training folds only.** Medians and MADs (and, for the comparison arm, means and standard deviations) are computed on each fold's training images and applied to its held-out images.
6. **No augmentation.** Jitter, rotation or elastic distortion would put the model's ideas about handwriting into the data this phase is meant to describe.

## Models and training

Two hidden-unit types are the core comparison, matching Nair & Hinton; a third is optional. All use Gaussian visible units with variance fixed at 1 and noise-free reconstructions.

| Model | Hidden unit | Hidden biases | Role |
| --- | --- | --- | --- |
| GB | Stochastic binary | yes | Baseline unit Nair & Hinton compared against |
| GR | Noisy rectified: max(0, x + N(0, σ(x))) | yes | Rectified unit as usually deployed |
| GR0 | Noisy rectified | none | Nair & Hinton's zero-bias condition, under which intensity equivariance holds |
| GL (optional) | Noisy leaky rectified | yes | Bridge to the Leaky ReLU phase; no published training recipe, so treated as unvalidated |

GR versus GR0 tests the founding claim directly: scale every test image by several factors above 0 and check whether each hidden unit's on/off state stays fixed.

**Training settings.** Starting values follow Hinton (2010) and Nair & Hinton (2010); any change made during the pilot is recorded with its reason.

| Setting | Starting value | Source |
| --- | --- | --- |
| Learning rule | Contrastive divergence, 1 step (CD-1) | Hinton 2010 |
| Learning rate | 1e-3 on the mean per-case update | Nair & Hinton 2010 |
| Momentum | 0.5 for the first 5 epochs, then 0.9 | Hinton 2010 |
| Weight decay | L2, 1e-4, weights only | Hinton 2010 |
| Initial weights | Normal, mean 0, SD 0.01 | Hinton 2010 |
| Initial biases | 0 | Hinton 2010 |
| Mini-batch | 10 images | Hinton 2010 (guide range 10–100); low end |
| Epochs | Up to 300; stop when the held-out free-energy gap widens | Nair & Hinton 2010; Hinton 2010 |
| Hidden units | 64, 128, 256 | Exploratory: compared to see how the features change (split, duplicate, go silent) as capacity grows. Nair & Hinton's 4,000 was fitted to their task, not carried over |

**Reference implementation.** A first implementation exists: a Gaussian-visible RBM with noisy rectified hidden units, written by DeepSeek in conversation with Phinn (`post-hair-pulling.py`). It has the training diagnostics built in: reconstruction error and gradient norm per epoch, hidden-unit activity, and original-beside-reconstruction images saved every 10 epochs. It uses no RBM library, and every line is inspected before it is used. It is the starting point, not the specification: where it differs from this protocol is listed below, and each difference is reconciled, or kept as a pilot arm, before Stage 1.

| Item | Reference script | This protocol | Status |
| --- | --- | --- | --- |
| Contextual data | Teklia validation split; bigrams cut by character-width proportion | HWDB2.1 DGRL lines, hand-cropped | Loader replaced; settled by the protocol |
| Seeds | Fixed at 42 | random.org, recorded and held | Replaced; settled |
| Held-out images | Reconstruction images drawn from training images | Writer-grouped folds; reconstructions read on held-out images | Replaced; settled |
| Hidden-unit noise | Fixed SD 0.01 | max(0, x + N(0, σ(x))), Nair & Hinton | To reconcile. At SD 0.01 the units are close to deterministic, the property the noise exists to avoid |
| Visible scaling | Pixels divided by 255; visible SD fixed at 1 | Standardised to unit variance (ledger step 10) | To reconcile. On 0–1 pixels, a visible SD of 1 is wider than the whole pixel range |
| Negative-phase visible units | Sampled: mean plus Gaussian noise | Noise-free (mean) | To reconcile, together with visible scaling |
| Free energy | Softplus hidden term, the binary-unit form | Used to watch for memorisation | Needs the rectified-unit form for GR and GR0; log1p(exp(x)) also overflows for large inputs |
| Resolution | 100 × 100 (10,000 visible units) | 32 × 32 primary, 48 × 48 arm | Open: add as a pilot arm or reconcile |
| Hidden units | 1,024 | 64, 128, 256 | Open |
| Mini-batch, epochs, momentum | 32; 100 fixed; 0.9 throughout | 10; until the held-out free-energy gap widens; 0.5 then 0.9 | Open |
| Gradient clipping | Norm clipped at 5 | Not in Nair & Hinton or Hinton (2010) | Kept as a logged training modification (Liao et al., 2022) |
| Polarity | Ink dark only, as CASIA stores it | Both polarities in the pilot | Add the ink-bright arm |
| Conditions | Study 2 (的) only | All four | Extend |
| Error summaries | Mean squared error over all pixels; epoch values averaged across mini-batches | Median absolute error, ink and background pixels separately; medians across mini-batches | Replaced; settled |

The script's header says it follows the Gaussian–Bernoulli RBM stability literature. Liao et al. (2022), *Gaussian-Bernoulli RBMs Without Tears*, propose a Gibbs–Langevin sampler and a modified contrastive divergence that lets the model generate from noise, and report that modified CD with gradient clipping trains these models robustly at large learning rates. The script takes only the clipping. The citation reached this project dated 2024; the arXiv paper is dated October 2022. DeepSeek also suggested starting values from a facial-keypoint thesis it did not identify (learning rate 0.005 for the Gaussian layer, momentum 0.9, 50 epochs); these are recorded as suggestions, not as a source.

## Instrument validation

No CASIA image is used until the RBM implementation passes four checks, the same gate the linear probe passed on MNIST.

1. **Positive control: MNIST.** Train GB and GR on MNIST at 28 × 28 with 256 hidden units. Pass: filters show localized stroke and blob detectors, reconstructions are recognizable digits, and the activity-frequency histogram has no units that are always on or always off.
2. **Negative control: noise.** Train on Gaussian noise images matched to CASIA's pixel mean and variance. Pass: filters show no spatial structure, and held-out reconstruction is no better than reconstructing the per-pixel mean.
3. **Permutation check.** A fully connected RBM has no notion of pixel neighbours. Train on CASIA images with one fixed pixel permutation applied, starting from the unpermuted run's initial weights permuted the same way, with the same batch order. Pass: the un-permuted filters match the unpermuted run's filters within floating-point tolerance. A mismatch means the code is leaking spatial structure it should not have.
4. **Fit monitor check.** Reconstruction error is not likelihood (Hinton 2010). Track the median free energy of training versus held-out images each epoch, with their interquartile ranges, alongside reconstruction error. Pass: on MNIST the free-energy gap widens when the model is deliberately overtrained on a 500-image subset.

A failed check stops the phase until the implementation is fixed and the check rerun.

## Design

The design is fully crossed and matched on seed: one seed produces the same initial weights for every condition and unit type, so differences in any output trace to the data or the unit, not the starting point.

**Stage 0, getting it to train.** Between the instrument checks and the pilot, the reference implementation is run on one condition (S1-ISO, since Study 1 runs first) and adjusted until its reconstructions hold structure worth describing. Gaussian-visible RBMs are known to be hard to train, so this stage may take many attempts. Every attempt is kept: its settings, curves and reconstruction images, and what was changed next and why. A run that produces noise, uniform grey or blobs is a result about training this model on this data and is reported as one; it is not discarded, and it is not read as a finding about isolated versus contextual data. The second condition is trained once the first reconstructs.

**Stage 1, pilot.** Settles the open preprocessing choices before the full run.

- Conditions S1-ISO and S1-CTX; units GB and GR; 128 hidden units; 3 seeds; fold 1 only.
- Crossed with polarity (2), scaling (mean/SD versus median/MAD) and resolution (32 versus 48).
- Output: the training curves, filters and residual maps for each choice, and a written record of which choice is kept and why.

**Stage 2, full run.**

| Factor | Levels | Count |
| --- | --- | --- |
| Condition | S1-ISO, S1-CTX, S2-ISO, S2-CTX | 4 |
| Hidden unit | GB, GR, GR0 | 3 |
| Hidden size | 64, 128, 256 | 3 |
| Seed | Drawn once from random.org (true random generator), recorded, and held; shared across all cells | 10 |
| Fold | 1 to 5 | 5 |
| **Total trained models** |  | **1,800** |

Ten seeds are enough to describe how stable each output is, which is the purpose here; this phase draws no inferential conclusions that would need more. GL, if run, adds 600 models.

**Folds.** Five folds, assigned once with their own seed (drawn from random.org and recorded, like the model seeds) and reused for every model. Folds are grouped by `writer_id` for both sources, so no writer appears in both a training and a held-out fold. Because the same characters can appear in different sentences, a sensitivity run also keeps each set of `neighbour_chars` within one fold. The crop holds only the pixels in its box, so this tests directly whether shared local characters let the model reconstruct by recall; if the held-out results do not change, that supports the claim that context cannot be memorized. Held-out images are used only to measure reconstruction and free energy on unseen data.

## Outputs and measurements

Eight outputs, each tied to a question, all descriptive. None uses a probe or reports an accuracy.

| Output | Answers | What is produced |
| --- | --- | --- |
| O1 Filter atlas | Q1 | Every hidden unit's weights drawn as an image, plus two subsets: top 36 by weight norm and 36 drawn at random |
| O2 Activity profile | Q2 | Histogram of `activity_freq` per model, overlaid by condition and unit type, and split by class |
| O3 Residual maps | Q3 | Per image: original, reconstruction, signed residual. Per class: median absolute residual map |
| O4 Hidden-code geometry | Q4 | Within one model: pairwise distances between class medians in its hidden-code space, including the same character isolated versus in a bigram |
| O5 Cross-source reconstruction | Q5 | Residuals when an ISO-trained model reconstructs CTX images and the reverse, against each model's own held-out residual |
| O6 Gate decomposition | Q6 | For GR and GR0: each held-out image rebuilt through the model's own transposed weights from three versions of its hidden layer (the on/off mask, the rectified output, and the full pre-activation including values below zero), shown beside the original |
| O7 Stability and fit | all | Seed-to-seed filter matching; training and held-out free energy and reconstruction curves; weight and bias histograms |
| O8 Rebuilt images | Q7 | Per held-out image: the original beside its median-of-draws reconstruction from the ISO-trained and the CTX-trained model of the same study, unit type, size and seed. Two quantities measured on the reconstruction, not the hidden layer: residual error by class, and the class of the nearest training image in pixel space |

**Notes on three outputs:**

- **O1.** Nair & Hinton showed the 216 filters with the highest weight norm and noted only about 25% of all filters were Gabor-like. Sorting by norm is a selection, so the random subset and the full atlas are always shown alongside it.
- **O4 and O5.** Run within a study only, since the two studies use different character sets. O5 restates the Study 1 transfer collapse (91.4% in-distribution, 5.26% on CalliBench) as a reconstruction question with no label in the way.
- **O6.** Uses no probe and no label as a target. What positive magnitude adds is seen as the difference between the mask and rectified reconstructions; what negative magnitude adds, as the difference between the rectified and pre-activation reconstructions, each shown as images and residuals. The pre-activation reconstruction sends values back through the weights that the model never sends back itself, so it shows what the gate removed, not something the model would produce.
- **Reconstruction type.** Rectified units are noisy, so a reconstruction is not one fixed image. Three kinds are labelled wherever shown: noise-off (hidden noise switched off; one pass back through the weights, giving the model's output b + Wh before any visible noise), median of draws (the pixel-wise median over a recorded number of hidden-noise draws, with the pixel-wise interquartile range shown beside it), and sampled (one draw, what the model actually generates). Median of draws is the primary view of what survived; one sampled reconstruction is shown beside it.

**打 in isolation versus in bigrams: reading it as EDA, not a disguised test.** The comparison is read from outputs O1, O2 and O4 already defined above, with no new transformation. What keeps it exploratory is the order of looking, not the outputs themselves. It becomes a hypothesis test the moment an expected pattern is written down first and then sought, or a threshold decides what counts.

1. **O1, filters.** Describe each model's full atlas and its random subset before looking at any target-specific question, and record what is there, including filters that fit no category. Only then note whether ISO filters resemble single stroke fragments and whether CTX filters span the boundary between two characters. A finding that neither holds is recorded the same way as one that does.
2. **O2, activity.** Each model is described first on its own training source, by class. Asking whether a unit fires for 打 in isolation versus in a bigram means passing images from the other source through a model, which is O5's cross-source input: the images differ in segmentation and intrusion as well as context, so any difference is read as a combined effect, not a context effect.
3. **O4, hidden codes.** Distances are taken inside one model, never between models. Two separately trained RBMs have unrelated hidden units (unit 12 in one has no counterpart in the other; Studies 1 and 2 found 0 of the top 20 channels shared), so the median of ISO-model codes and the median of CTX-model codes are not in the same space. Instead, each model in turn encodes both isolated 打 and bigram 打, and the distance between those two medians is compared with the distances between 打 and the distractors in that same model.

Whatever is seen is written up as description: what was looked at, in what order, and what was there.

## Analysis and reporting rules

Results are reported as distributions across seeds and folds, never as a single run, and nothing is tested for significance.

- **Summaries.** Median, interquartile range and full range. No means. Anything summarised across the pixels of one image is summarised separately over ink and background pixels, because background is most of every image and a median over all pixels would describe the paper, not the writing.
- **Counts.** Absolute numbers of units, images and models, with denominators. Percentages only alongside the counts they come from.
- **Pictures first.** Every summary number has the picture it summarizes: the atlas, the histogram or the residual map.
- **Seed stability.** A pattern is described as stable only if it appears in at least 8 of 10 seeds; otherwise the count is reported as is.
- **Selection.** Any subset shown (units, images, classes) states how it was chosen, and a random subset of the same size is shown beside it.
- **Pilot decisions.** Each preprocessing choice from Stage 1 is written down with the evidence that decided it, before Stage 2 starts.
- **Provenance.** All seeds (model initialization and fold assignment) are drawn from random.org's true random generator, recorded with the date drawn, and held fixed for reproducibility. Every run logs the code commit, environment snapshot, seed, fold and ledger settings, and terminal output is captured to file from the first command.
- **Failed runs are kept.** A run that does not train, or gives an unexpected picture, is reported with the rest. Nothing is dropped for looking wrong.
- **AI assistance.** DeepSeek wrote the reference implementation and suggested starting values. Claude (Anthropic) helped draft this protocol and wrote the DGRL parser. Neither is cited as the source of a claim; each claim is traced to a paper, the data or a run.

## Decision points for the next phase

The EDA is finished when each observation below has been checked, and what it shows decides the next design. These are readings to look for, not predictions.

| If the EDA shows | Then the next phase |
| --- | --- |
| CTX filters span the boundary between two characters | Needs representations wider than one character; per-character crops would discard the signal |
| Residuals concentrate where older writing or neighbouring lines intrude | Revisit cropping and intrusion handling before any model comparison |
| Fine stroke detail is lost at 32 pixels | Raise resolution or use convolutional RBMs, which keep spatial structure |
| GR and GR0 differ in on/off stability under intensity scaling | Hidden biases break the equivariance property; the Leaky ReLU phase must report biases as a factor |
| In O6, pre-activation reconstructions show structure the rectified ones lack, for CTX but not ISO | Supports the Leaky ReLU protocol's information-loss arm; run it first |
| In O6, pre-activation and rectified reconstructions look alike in both conditions | Points to pooling rather than the gate as the larger loss in the CNN; test pooling directly before the Leaky ReLU arms (the RBM has no pooling, so this is a lead, not a finding) |
| Filters or activity profiles are unstable across seeds | The data does not constrain a single solution; any later comparison needs the distributional design |
| Reconstructions are still noise, grey or blobs after Stage 0 | The Stage 0 record is reported as the finding about this model on this data; the next phase changes the instrument (for example the Without Tears sampler or a convolutional RBM) rather than reading the data through a model that did not train |

Whatever is observed is written up as the description of the data's shape, with its pictures, before the next protocol is drafted.

## Known limitations

This phase makes no inference, so these are not threats to one. They are what the instrument and the handling of the data contribute to what can be seen, recorded so that a pattern is not mistaken for the data's own shape when part of it came from the tool. The RBM was chosen because it adds little: no labels, no task, no notion of similarity, no spatial prior. What it does add is listed first.

**What the RBM itself contributes**

- **Learning rule.** CD-1 does not follow the true likelihood gradient, so filters reflect the data and the learning rule together. Comparing GB, GR and GR0 on the same data shows which patterns move with the rule and the unit type, and which stay with the data.
- **Hidden-unit type.** Binary and rectified units impose different limits on what a unit can express. That is the comparison this phase is built around, and it is also a property of the tool: patterns seen under one unit type only are reported as such.
- **Stochastic hidden units.** GR samples noise during training; every reconstruction is labelled noise-off, median of draws, or sampled (Outputs: Reconstruction type). Gradient clipping and any change made in Stage 0 are also part of the tool, and are logged with the run.
- **No likelihood, by design not a goal.** Free energy and reconstruction are not used to judge how well the RBM fits or to rank models. They serve one purpose: to show when a model has started copying its training images instead of describing structure, the point at which training stops.

**What the data and its handling contribute**

- **Complexity, not count.** Each condition holds the images the source contains. What makes this data demanding is the variation inside it (writer, stroke, neighbouring characters, how CASIA segmented it), not how many images there are. No choice in this protocol is set by image count, and nothing is treated as more or less trustworthy because of it. Memorisation is watched for directly, through the held-out free-energy gap, rather than assumed from a count.
- **Different segmentation.** Isolated and contextual images share CASIA's capture process, grey format and writer pool (the HWDB1.1/2.1 batch), but CASIA segmented one by character and the other by line, and contextual crops carry intrusion from adjacent lines. The ledger records these; differences seen between ISO and CTX are read as including them.
- **Writer structure.** All four conditions are drawn from one writer pool: isolated images from HWDB1.1 and contextual images from HWDB2.1, which CASIA's documentation states were written by the same 300 writers. This holds for both studies, since Study 2's 的 images were also read from the HWDB1.1 training GNT files. Matching is at the pool level until the GNT and DGRL writer numbers are confirmed to align, and the contextual arm covers only the writers whose text pages contain 打 or 的. Some structure the RBM finds may be writer style rather than character; `writer_id` lets this be seen.
- **No spatial prior, and what follows from it.** A fully connected RBM treats pixels as unrelated to their neighbours, which is why it suits this phase. It also means what it shows describes the images, not what a CNN, which builds that prior in, would make of them.
- **Unofficial copy.** HWDB2.1 was obtained from a Hugging Face re-upload, not from CASIA. Its integrity rests on the parser's byte accounting and on checking lines by eye until it is compared with an official checksum.
- **Human coding.** `intrusion` and every crop box are set by one person, with no second rater in this phase.
- **Looking is an instrument too.** Reading reconstructions by eye is a measurement with a frame: one observer, human vision, and the rendering chosen (grey scale, contrast range). The frame is stated with every figure, the pictures are shown so any reader can look for themselves, and the two image-side quantities in O8 sit beside them.

## References

- Nair, V. & Hinton, G. E. (2010). Rectified linear units improve restricted Boltzmann machines. *ICML 2010*. Read in full for this protocol.
- [Alain, G. & Bengio, Y. (2016, v4 2018). Understanding intermediate layers using linear classifier probes. arXiv:1610.01644](https://arxiv.org/abs/1610.01644). Read in full; source of the probe procedure validated in Study 2.
- Hinton, G. E. (2010). A practical guide to training restricted Boltzmann machines. University of Toronto, UTML TR 2010-003. Not reopened while drafting; training values to be checked against it before the pilot.
- Liu, C.-L., Yin, F., Wang, D.-H. & Wang, Q.-F. (2011). CASIA online and offline Chinese handwriting databases. *ICDAR 2011*. Primary citation for both datasets. Not reopened while drafting; needed for ledger step 1. The HWDB1.1/2.1 writer pairing was taken from CASIA's dataset documentation.
- [OrkaZeta/HWDB2](https://huggingface.co/datasets/OrkaZeta/HWDB2). Hugging Face re-upload of the HWDB2.0–2.2 DGRL archives; source of the contextual copy used here.
- Lee, H., Grosse, R., Ranganath, R. & Ng, A. Y. (2009). Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations. *ICML 2009*. Not reopened while drafting; cited for convolutional RBMs.
- Liao, R., Kornblith, S., Ren, M., Fleet, D. J. & Hinton, G. (2022). Gaussian-Bernoulli RBMs without tears. arXiv:2210.10318. Abstract read; source of the gradient-clipping modification.
- Han, S., Mao, H. & Dally, W. J. (2016). Deep compression: compressing deep neural networks with pruning, trained quantization and Huffman coding. ICLR 2016; arXiv:1510.00149. Informed the Huffman framing. Not reopened while drafting.
- Glorot, X., Bordes, A. & Bengio, Y. (2011). Deep sparse rectifier neural networks. AISTATS 2011. Not reopened while drafting; cited for the uptake of rectified units in feed-forward networks.
- Jarrett, K., Kavukcuoglu, K., Ranzato, M. & LeCun, Y. (2009). What is the best multi-stage architecture for object recognition? ICCV 2009. Not reopened while drafting; cited for earlier use of rectification.
