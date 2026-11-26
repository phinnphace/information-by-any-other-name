# RBM Exploratory Protocol: Learning the Shape of the Data

Oct 17, 2026 · @phinn markson

## Purpose and stance

This phase learns the shape of the CASIA data from the data, before any task, label or architecture tells it what to look for. It is an exploratory data analysis, run with a generative model instead of summary plots.

Every model in Studies 1 and 2 had an objective that defined what mattered: a binary label, a six-way label, or a contrastive definition of similar. Each objective is a lens, and the results described the data as seen through that lens. A restricted Boltzmann machine (RBM) models p(v), the distribution of the images themselves. It has no labels and no imposed notion of similarity, so whatever structure it finds came from the images.

**How this departs from standard practice:**

- No accuracy, no benchmark, no train/test performance claim. Labels are used only after training, to stratify descriptive outputs.
- Questions, not hypotheses. Nothing here is tested for significance; the outputs are looked at, described and counted.
- The instrument is validated before it is trusted (Section: Instrument validation), as the linear probe was in Study 2.
- Every transformation between the handwritten page and the model's input is logged with its known or unknown distance from the observation (Section: Data dictionary, part 1). This step is building on prior work where I elucidate data transformations across administrative boundaries operate under a variable agnostic pertubration that is a proprty of the transormation, and distance is equal to proxy - direct observation amplifed by ignorance of conflation of the former. 

**What this phase will not claim:** that one pretraining regime, unit type or data source is better. Its output is a description of the data's structure and of what each hidden-unit type keeps and discards. That description sets the design of the next phase.

## Background

Studies 1 and 2 left a contradiction that no task-driven model can resolve: isolated pretraining always produced sparser conv3 representations, yet the better model flipped between characters.

| Measure | Study 1 (打) | Study 2 (的) |
| --- | --- | --- |
| Isolated model (A) | 61.2% | 90.4% (relu3 probe) |
| Contextual model (B) | 91.4% | 66.3% (relu3 probe) |
| Better model | B | A |
| conv3 fraction negative, A | 0.844 | 0.992 |
| conv3 fraction negative, B | 0.780 | 0.854 |
| Top-20 channel overlap, A vs B | 0 of 20 | 0 of 20 |

Study 2 also showed the contextual model's character signal was present before the gate: a pre-ReLU spatial probe reached 80.8% where the post-ReLU pooled probe reached 66.3%. Across 1,000 matched seeds the A − B gap at relu3 had a 95% interval of −0.092 to 0.379, with a sign flip on 68 seeds. Sparsity is a correlate of the pretraining source, not the mechanism that decides which model wins.

**Why an RBM, and why this one.** Several reasons. The Ml + paper mill crisis timeline. (https://cacm.acm.org/practice/research-for-practice-12/) leads me to have to go back in time to 'pre-paper mill at scale' Pre-paper mills at scale generate the conflated results, papers, and problematic methodologies. Going back to an earlier paper with a well-written and reproducible methodology, I can reproduce their methodology as science intends. Initially, the plan was to use LeakyReLu to gain more specific information about the qualit, location and type of information loss at the ReLu3 gate.

However, due to the aformentioned methodological issues with scholarly literature and further review, I found this earlier work by Nair & Hinton (2010). This methodology will allow for a more honest, scientifically rigorous exploratory data analysis.  Nair & Hinton (2010) introduced rectified hidden units into RBMs and made the claim later cited as "ReLUs preserve information." Read closely, that claim holds only for zero-bias units, only relative to binary units, and only for units above zero. The paper also gives the diagnostics this phase reuses: learned filters shown as images, and a histogram of how often each hidden unit is active (their Figure 5), which differed between NORB and LFW with the same unit type.

An RBM reconstructs its input. That makes the distance between representation and observation directly visible as a per-pixel residual, which no classifier provides. Hinton's practical guide (2010) is the operating manual for training, and it warns that reconstruction error is a misleading measure of fit. That warning is treated here as an instrument-val idation requirement, not a footnote and buttresses prior work that I have been exploring in other domains. If something is true it is true everywhere. I speak of axiomic truth, not fragile findings, results, or environment/domain specific 'results'.

## Exploratory questions

Six questions frame what to look at. None carry a predicted answer, and none are tested for significance.

1. **Q1 Structure.** What do the hidden units learn from each data source: stroke fragments, radicals, whole characters, the neighbouring character, or the older writing underneath?
2. **Q2 Activity profile.** How often is each hidden unit active, and does that profile differ by data source, unit type and character?
3. **Q3 Where reconstruction fails.** Which pixels and regions does the model consistently fail to hold, per character and per source?
4. **Q4 Same character, two settings.** Does 打 (or 的) in isolation land near 打 in a bigram in hidden-code space, or do the two occupy separate regions?
5. **Q5 Cross-source reconstruction.** How well does an RBM trained on one source reconstruct images from the other? This is the transfer question restated in generative form.
6. **Q6 What the gate discards.** For rectified units, what do the hidden pre-activations below zero carry, separated into sign pattern, positive magnitude and negative magnitude?

## Data sources and conditions

One RBM is trained per data condition, on all classes pooled, because the model is unsupervised. Labels are attached afterwards for stratified description only.

| Condition | Study | Source | Content | Images |
| --- | --- | --- | --- | --- |
| S1-ISO | 1 | CASIA HWDB1.1, `data/isolated/` | 打 plus 扔 扛 扫 托 扣 | 489 (239 打, 50 each distractor) |
| S1-CTX | 1 | CASIA-HWDB2 lines, `data/condition_b/` | 打 bigram crops plus negative bigrams, hand-cropped during preprocessing | 286: 151 打 (every occurrence in 147 validation images) + 135 negatives (random sample); about 53/47 by design |
| S2-ISO | 2 | CASIA HWDB1.1 GNT files | 的 plus 地 得 也 他 她 | 1,437 (≈240 per class) |
| S2-CTX | 2 | CASIA-HWDB2 validation split | 的 bigram crops plus negative bigrams, proportional crop | 300 (150 + 150) |

Study 1 is run first, because the original hypothesis held there and its bigrams were hand-cropped rather than cut by proportion. Study 2 follows with the same pipeline.

Open items to confirm against the files before running:

- [ ] Image file format and bit depth for the Study 1 crops
- [ ] Colour mode of every crop file (a sample shows colour PNGs named by source line, position and text window)

## Data dictionary, part 1: the proxy ledger

The image a model sees is several transformations away from the handwriting. Each row below is one transformation, with what it changes and whether that distance is known.

The ledger exists because an unmeasured distance cannot be bounded, and one unbounded term makes the total unstateable:

```latex
E_{total} \approx \sum_i d_i \cdot A(k_i, n_i)
```

Here d is the distance a transformation introduces, k whether that distance is known, and n how many later steps depend on it.

| Step | Transformation | What it changes | Delta status | How it gets measured |
| --- | --- | --- | --- | --- |
| 1 | Page scanned by CASIA | Resolution, intensity quantization | Unknown | Confirm acquisition settings in the CASIA dataset paper (Liu et al., 2011) |
| 2 | Character segmented by CASIA (HWDB1.1) | Crop boundary, background set to white | Known, unmeasured | Record each GNT image's original width and height |
| 3 | Source images (HWDB2), each holding anywhere from a bigram to most of a paragraph | No consistent unit of text per image; strokes from surrounding text can intrude | Unknown | Visual coding of intrusion per crop (Part 2, `intrusion`) |
| 4 | Bigram cropped by character-width proportion (Study 2) | Crop edges assume equal-width characters | Known, unmeasured | Compare a manually corrected subset against the proportional crop |
| 5 | Bigram hand-cropped to 100 × 100 during preprocessing, before any study (Study 1 data) | Per-image human judgement; older writing left in place | Known, unmeasured | Describe residuals separately by crop method |
| 6 | Resized to 100 × 100, Lanczos filter | Aspect ratio forced to square; interpolation | Measured in this phase | Original aspect ratio per image vs reconstruction residual |
| 7 | Converted to one grey channel and divided by 255 at load time (Studies 1 and 2 loaders) | Hue and paper tone discarded; contextual crops are stored in colour, isolated GNT images are grey | Known, unmeasured | Pilot runs contextual crops in colour and in grey (Preprocessing, step 3) |
| 8 | Downsampled 100 × 100 to 32 × 32 for the RBM | Fine stroke detail | Measured in this phase | Down-then-up residual per image, before any training |
| 9 | Polarity set (ink dark or ink bright) | Which pixels are near zero; interacts with zero-bias rectified units | Measured in this phase | Run both polarities in the pilot |
| 10 | Per-pixel mean removed, divided by one global standard deviation | Where zero sits for every hidden unit's input | Measured in this phase | Compare against median/MAD scaling in the pilot |

## Data dictionary, part 2: record variables

The dictionary records provenance and transformations, not presumed features. What the images contain is the output of this phase, so no feature is defined in advance.

**Input records, one row per image:**

| Variable | Type | Values | Meaning |
| --- | --- | --- | --- |
| `image_id` | string | unique | Stable identifier across all runs |
| `condition` | category | S1-ISO, S1-CTX, S2-ISO, S2-CTX | Data condition |
| `class_label` | category | the six characters per study | Used after training only |
| `is_target` | boolean | true, false | 打 or 的 present |
| `writer_id` | string | GNT file name | Isolated images only: each HWDB1.1 GNT file is one writer, so this comes free |
| `source_line` | string | from file name, e.g. `line_0127` | Contextual images only: the source image a crop was cut from |
| `line_position` | integer | from file name, e.g. `pos_3` | Position of the target character in its source image |
| `text_window` | string | from file name, e.g. `text_得上打胜仗` | The characters around the target; the same characters can appear at different positions, in different sentences |
| `source_file` | string | path | GNT file or crop file |
| `colour_mode` | category | RGB, RGBA, L | Stored colour mode of the file |
| `orig_width`, `orig_height` | integer | pixels | Size before resizing (ledger step 6) |
| `crop_method` | category | proportional, hand-cropped | Ledger steps 4 and 5 |
| `intrusion` | category | none, surrounding text, older writing, both | Coded by visual inspection |
| `downsample_residual` | float | 0 and up | Mean absolute down-then-up error (ledger step 8) |
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

Per-image outputs (reconstruction residual, hidden code, free energy) are stored keyed on `image_id` and `model_id`.

## Preprocessing

Preprocessing is kept to the minimum the RBM needs, and every step that is a choice is run both ways in the pilot rather than fixed by convention.

1. **Inventory before transforming.** Record `orig_width`, `orig_height`, `writer_id` or `source_line`, and `crop_method` for every image. Code `intrusion` by eye for every contextual crop.
2. **Resolution.** Primary resolution is 32 × 32 (1,024 visible units), following Nair & Hinton. Before training, compute each image's down-then-up residual at 32 and at 48 pixels and report its distribution per condition and class. If fine strokes that separate characters are lost at 32, the 48-pixel run becomes primary.
3. **Polarity.** Run ink-dark and ink-bright in the pilot. Polarity is not lossy, but it decides which pixels sit near zero, which matters to rectified hidden units. Colour: contextual crops are stored in colour and isolated images are grey. The pilot runs contextual crops in colour (3,072 visible units at 32 × 32) and in grey, with isolated images as grey repeated to three channels, so the asymmetry is explicit rather than hidden.It's again much of the information. Nicki is eating more information. Information, yes, introduced and rectified in RVMs. I apologize. That claim holds for zero. I feel through some out. No. Stop this. Trash. That's a yes. I like that. I thought I'd fix this, but apparently not. What the fuck? Huh. Next work. Huh. Yes. Okay. Weird, fuck. Huh. Um, where'd you go then? Stop this. 
4. **Scaling.** Primary: subtract the per-pixel mean and divide by one global standard deviation (Gaussian visible units with unit variance, per Hinton 2010). Sensitivity: per-pixel median and median absolute deviation. Nair & Hinton's wording leaves per-pixel versus global centring ambiguous, so the choice is stated here.
5. **Fit on training folds only.** Means, standard deviations, medians and MADs are computed on each fold's training images and applied to its held-out images.
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
| Mini-batch | 10 images | Hinton 2010, given 286 to 1,437 images |
| Epochs | Up to 300; stop when the held-out free-energy gap widens | Nair & Hinton 2010; Hinton 2010 |
| Hidden units | 64, 128, 256 | Data sizes rule out Nair & Hinton's 4,000 |

The implementation is written from scratch in PyTorch, about 100 lines, with no RBM library. Every line of it is inspected before it is used.

## Instrument validation

No CASIA image is used until the RBM implementation passes four checks, the same gate the linear probe passed on MNIST.

1. **Positive control: MNIST.** Train GB and GR on MNIST at 28 × 28 with 256 hidden units. Pass: filters show localized stroke and blob detectors, reconstructions are recognizable digits, and the activity-frequency histogram has no units that are always on or always off.
2. **Negative control: noise.** Train on Gaussian noise images matched to CASIA's pixel mean and variance. Pass: filters show no spatial structure, and held-out reconstruction is no better than reconstructing the per-pixel mean.
3. **Permutation check.** A fully connected RBM has no notion of pixel neighbours. Train on CASIA images with one fixed pixel permutation applied, starting from the unpermuted run's initial weights permuted the same way, with the same batch order. Pass: the un-permuted filters match the unpermuted run's filters within floating-point tolerance. A mismatch means the code is leaking spatial structure it should not have.
4. **Fit monitor check.** Reconstruction error is not likelihood (Hinton 2010). Track the mean free energy of training versus held-out images each epoch alongside reconstruction error. Pass: on MNIST the free-energy gap widens when the model is deliberately overtrained on a 500-image subset.

A failed check stops the phase until the implementation is fixed and the check rerun.

## Design

The design is fully crossed and matched on seed: one seed produces the same initial weights for every condition and unit type, so differences in any output trace to the data or the unit, not the starting point.

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
| Seed | drawn from os.urandom, shared across all cells | 10 |
| Fold | 1 to 5 | 5 |
| **Total trained models** |  | **1,800** |

Ten seeds are enough to describe how stable each output is, which is the purpose here; this phase draws no inferential conclusions that would need more. GL, if run, adds 600 models.

**Folds.** Five folds, fixed once with their own seed and reused for every model. Isolated folds are grouped by `writer_id`, so no writer appears in both a training and a held-out fold. Contextual folds are grouped by `source_line`, read from the crop file names. Because the same `text_window` can appear in different sentences, a sensitivity run keeps each text window within one fold. The crop holds only the pixels in its box, so this tests directly whether shared local characters let the model reconstruct by recall; if the held-out results do not change, that supports the claim that context cannot be memorized. Held-out images are used only to measure reconstruction and free energy on unseen data.

## Outputs and measurements

Seven outputs, each tied to a question, all descriptive. Only O6 reports an accuracy, and there it measures what information a layer holds, not how well a model performs.

| Output | Answers | What is produced |
| --- | --- | --- |
| O1 Filter atlas | Q1 | Every hidden unit's weights drawn as an image, plus two subsets: top 36 by weight norm and 36 drawn at random |
| O2 Activity profile | Q2 | Histogram of `activity_freq` per model, overlaid by condition and unit type, and split by class |
| O3 Residual maps | Q3 | Per image: original, reconstruction, signed residual. Per class: median absolute residual map |
| O4 Hidden-code geometry | Q4 | Pairwise distances between class medians in hidden-code space; same-character distance, isolated versus bigram |
| O5 Cross-source reconstruction | Q5 | Residuals when an ISO-trained model reconstructs CTX images and the reverse, against each model's own held-out residual |
| O6 Gate decomposition | Q6 | For GR and GR0: probes on the hidden on/off mask, the rectified output and the pre-activation |
| O7 Stability and fit | all | Seed-to-seed filter matching; training and held-out free energy and reconstruction curves; weight and bias histograms |

**Notes on three outputs:**

- **O1.** Nair & Hinton showed the 216 filters with the highest weight norm and noted only about 25% of all filters were Gabor-like. Sorting by norm is a selection, so the random subset and the full atlas are always shown alongside it.
- **O4 and O5.** Run within a study only, since the two studies use different character sets. O5 restates the Study 1 transfer collapse (91.4% in-distribution, 5.26% on CalliBench) as a reconstruction question with no label in the way.
- **O6.** The only output using labels as a target. The difference between the mask probe and the rectified probe estimates what positive magnitude adds; the difference between the rectified and pre-activation probes estimates what negative magnitude adds. Probes use the validated linear-probe procedure, fitted on training folds and scored on held-out folds.

## Analysis and reporting rules

Results are reported as distributions across seeds and folds, never as a single run, and nothing is tested for significance.

- **Summaries.** Median, interquartile range and full range. No means without the distribution shown beside them.
- **Counts.** Absolute numbers of units, images and models, with denominators. Percentages only alongside the counts they come from.
- **Pictures first.** Every summary number has the picture it summarizes: the atlas, the histogram or the residual map.
- **Seed stability.** A pattern is described as stable only if it appears in at least 8 of 10 seeds; otherwise the count is reported as is.
- **Selection.** Any subset shown (units, images, classes) states how it was chosen, and a random subset of the same size is shown beside it.
- **Pilot decisions.** Each preprocessing choice from Stage 1 is written down with the evidence that decided it, before Stage 2 starts.
- **Provenance.** Every run logs the code commit, environment snapshot, seed, fold and ledger settings, and terminal output is captured to file from the first command.

## Decision points for the next phase

The EDA is finished when each observation below has been checked, and what it shows decides the next design. These are readings to look for, not predictions.

| If the EDA shows | Then the next phase |
| --- | --- |
| CTX filters span the boundary between two characters | Needs representations wider than one character; per-character crops would discard the signal |
| Residuals concentrate where older writing or neighbouring lines intrude | Revisit cropping and intrusion handling before any model comparison |
| Fine stroke detail is lost at 32 pixels | Raise resolution or use convolutional RBMs, which keep spatial structure |
| GR and GR0 differ in on/off stability under intensity scaling | Hidden biases break the equivariance property; the Leaky ReLU phase must report biases as a factor |
| Negative magnitude adds information in O6 for CTX but not ISO | Supports the Leaky ReLU protocol's information-loss arm; run it first |
| Negative magnitude adds nothing in either condition | Points to pooling rather than the gate as the larger loss in the CNN; test pooling directly before the Leaky ReLU arms (the RBM has no pooling, so this is a lead, not a finding) |
| Filters or activity profiles are unstable across seeds | The data does not constrain a single solution; any later comparison needs the distributional design |

Whatever is observed is written up as the description of the data's shape, with its pictures, before the next protocol is drafted.

## Known limitations

The RBM is itself an instrument with its own transformations, and these are its known distances.

- **No likelihood.** The partition function is intractable, so how well the RBM models the data cannot be measured directly. Free energy and reconstruction are proxies for fit, and are labelled as such.
- **Biased learning rule.** CD-1 does not follow the true likelihood gradient. Filters reflect the data and the learning rule together.
- **No spatial prior.** A fully connected RBM ignores which pixels are neighbours. That suits an EDA, but it means findings do not transfer directly to a CNN, which builds that prior in.
- **Small samples.** 286 to 1,437 images per condition limit how many hidden units can be supported without memorization; the held-out free-energy gap is the guard.
- **Unequal sources.** Isolated and contextual images differ in segmentation, crop method and intrusion, not only in context. The ledger records these; it cannot remove them.
- **Writer structure.** Isolated classes share writers, so some structure the RBM finds may be writer style rather than character. `writer_id` lets this be checked for isolated images, not excluded. Contextual crops carry no writer metadata, so there it cannot be checked.
- **Stochastic hidden units.** GR samples noise during training; outputs computed from deterministic activations are a further approximation, stated where used.
- **Human coding.** `intrusion` is coded by eye by one person, with no second rater in this phase.

## References

- Nair, V. & Hinton, G. E. (2010). Rectified linear units improve restricted Boltzmann machines. *ICML 2010*. Read in full for this protocol.
- [Alain, G. & Bengio, Y. (2016, v4 2018). Understanding intermediate layers using linear classifier probes. arXiv:1610.01644](https://arxiv.org/abs/1610.01644). Read in full; source of the probe procedure validated in Study 2.
- Hinton, G. E. (2010). A practical guide to training restricted Boltzmann machines. University of Toronto, UTML TR 2010-003. Not reopened while drafting; training values to be checked against it before the pilot.
- Liu, C.-L., Yin, F., Wang, D.-H. & Wang, Q.-F. (2011). CASIA online and offline Chinese handwriting databases. *ICDAR 2011*. Not reopened while drafting; needed for ledger step 1.
- Lee, H., Grosse, R., Ranganath, R. & Ng, A. Y. (2009). Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations. *ICML 2009*. Not reopened while drafting; cited for convolutional RBMs.
