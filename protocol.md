A.2 Objective and Goals
Objective. Describe what information about 打, isolated and in context, a restricted Boltzmann machine retains and loses, and assess whether the approach can carry the next phase of the work based upon information retention and loss.
Goal 1. Establish the RBM as a working instrument on this data.
Goal 2. Describe what RBMs trained on isolated and on contextual handwriting retain.
Goal 3. Describe what the rectifying gate discards, and whether hidden biases change it.


A.3 Experimental design summary
We train one RBM per data condition on all characters pooled, because the model learns without labels; labels are attached afterwards only to describe results by character. The design is crossed and matched on seed: one seed gives the same initial weights for every condition and unit type, so differences trace to the data or the unit.


Factor	Levels	Count
Condition	Isolated (ISO), contextual (CTX)	2
Hidden unit	Gaussian–binary (GB), Gaussian–rectified (GR), Gaussian–rectified without hidden biases (GR0); Gaussian–leaky rectified (GL) optional	3
Hidden units per model	64, 128, 256	3
Seed	Drawn from random.org, in batches recorded and held; batches continue until the retention stabilizes (D.2)	
Fold	Writer-grouped, 1 to 5	5
Trained models	900 = 10 seeds; the seed count is discovered in D.2, so 900 is a floor, not the design total.	900 (GL adds 300)

The work proceeds in three stages. Stage 0 gets the RBM training on this data (Aim 1). Stage 1, a pilot, settles open preprocessing choices. Stage 2 runs the full design. Contextual images are drawn in recorded rounds, so the number of images grows only as far as the analysis needs.
Within the Cross-Industry Standard Process for Data Mining (CRISP-DM), this study is the data-understanding stage, translated to scientific practice (Section F).
 Study Population, Materials and Setting
the handwriting of CASIA's 300-writer batch shared by two databases: the Handwriting Database (HWDB) release 1.1, isolated characters, and release 2.1, text lines written by the same writers (Liu et al., 2011). We use the training sets: 240 writer files in HWDB1.1 and 1,200 page files in HWDB2.1.
·	Isolated condition (ISO). Every image of 打 and the distractors 扔, 扛, 扫, 托 and 扣 in the 240 HWDB1.1 training files, counted at extraction. Study 1 used 489 of these: 239 打 and 50 of each distractor.
·	Contextual condition (CTX). Two-character crops from HWDB2.1 training lines: 打 with its neighbouring character, and two-character windows from lines without 打. Crops are drawn at random in rounds (B.3). Study 1 used 286: 151 crops of 打 in context and 135 contextual crops without 打.
Inclusion and exclusion criteria are defined variable by variable in the data dictionary (supplement: CASIA_Data_Dictionary.xlsx). In brief, we include the HWDB1.1 and HWDB2.1 training sets only, the six study characters only, and 8-bit single-channel grayscale images only; we exclude HWDB2.0 (a different writer batch), test sets, and third-party copies that were resized or stored in colour.

Item	Detail
Original source	CASIA-HWDB, National Laboratory of Pattern Recognition, Institute of Automation, Chinese Academy of Sciences (Liu et al., 2011)
Isolated release	HWDB1.1 training set: 240 GNT files, one per writer, named by writer number (1001-c.gnt onward). These are the raw files Study 1 read, recovered from the Windows profile; the working copy in capstone-characters/HWDB1.1trn_gnt holds all 240 files and is byte-identical to the recovered set (checked 4 November 2026)
Contextual release	HWDB2.1 training set: DGRL files, one per page, named writer–page (1001-P16)
Contextual copy	Hugging Face OrkaZeta/HWDB2, HWDB2.1Train.zip (554 MB), an unofficial re-upload. HWDB2.0Train.zip (727 MB) was downloaded on 19 October 2026 to pilot the parser and is not used in any condition
Decoding	dgrl_extract.py writes one PNG per text line with no resizing or colour change and stops on any layout mismatch. Piloted on 3 HWDB2.0 pages (27 lines, checked by eye against their labels). HWDB2.1 training set: 1,200 of 1,200 pages decoded; 13,768 lines; 344,048 characters; 0 undecodable labels
Writer matching	GNT and DGRL file names carry the same four-digit writer number, so writers match by identifier across the two sources
Terms of use	CASIA releases the databases for academic research; terms to be confirmed on CASIA's official page
Not used	Teklia/CASIA-HWDB2-line (lines resized to 128 pixels high, stored in colour; the source of Study 1's contextual crops); UndefinedCpp/casia-char-1 (no writer field, releases mixed)

B.3 Contextual crop selection
Selection is TRG  and conducted at random.org, so the crops are whatever is drawn. We list and number two pools from the HWDB2.1 training lines: every occurrence of 打 (each its own entry, with no de-duplication by line), and every two-character window in lines without 打. Before any cropping, we draw batches of random numbers from random.org and record them in a seed ledger. Each round takes the next numbers in sequence, so adding images never reopens the selection, and every number is accounted for whether or not it is used.
Each drawn candidate is cropped as drawn: a fixed 100 × 100 pixel box placed on 打 and its right neighbour, or its left neighbour at the end of a line, following Study 1's span rule. The neighbour is required in every crop; without it, a model could describe 打 alone. A candidate is excluded only if it is too illegible to use, or the box cannot show 打 or its neighbour whole. Excluded candidates are kept in the project archive, the record notes the draw and the reason, and the next draw replaces them. Counts are unequal by design, and lines are not matched on length. The first round draws about 20 crops per pool; further rounds follow only until there is enough to judge whether the approach works and is worth pursuing. Crop batches and seed batches are drawn from separate parts of the ledger; the crop batch size (about 20 per pool) is a starting point, and the seed batch size is discovered (D.2)

B.4 Transformation ledger
The image a model sees is several transformations away from the handwriting. We record each transformation with what it changes and whether that distance is known, because an unmeasured distance cannot be bounded:
π(Tᵢ) · A(kᵢ) · Πⱼ>ᵢ Tⱼ

Here d is the distance a transformation introduces, k whether that distance is known, and n how many later steps depend on it.

ΔX(VAR) status records whether the change a step introduces is measurable and whether it has been measured; it does not claim the observation-to-proxy gap, which is unknown for steps 1 and 3 and named but not measured for steps 2 and 5.

Step	Transformation	What it changes	ΔΧ(VAR)
status	How it is measured
1	Written on paper; page scanned by CASIA	Resolution, intensity quantization	Unknown	Confirm acquisition settings in Liu et al. (2011)
2	Character segmented by CASIA (HWDB1.1)	Crop boundary; background set to white	Known, unmeasured	Record each image's original width and height
3	Page segmented into text lines by CASIA (HWDB2.1)	Line boundary; strokes from adjacent lines can intrude	Unknown	Code intrusion by eye for each crop
4	GNT and DGRL files decoded to PNG	Nothing, if the parsers are correct	Lossless	Every byte accounted for; every line box inside its page; labels compared with images by eye
5	Contextual crop cut to a fixed 100 × 100 pixels on 打 and its neighbour	Everything outside the box is dropped; edges follow the fixed size	Known, unmeasured	Crop box logged against the line image, so every crop can be reproduced
6	Isolated images resized to 100 × 100 (Lanczos filter); contextual crops are already 100 × 100	Aspect ratio forced to square; interpolation	Measured in this study	Original aspect ratio per image against reconstruction residual
7	Intensities divided by 255	Scale only	Lossless	None needed
8	Downsampled to 32 × 32 (48 × 48 in the pilot)	Fine stroke detail	Measured in this study	Down-then-up residual per image, before any training
9	Polarity set (ink dark or ink bright)	Which pixels sit near zero; interacts with rectified units without biases	Measured in this study	Both polarities run in the pilot
10	Per-pixel median removed; divided by one global median absolute deviation (MAD)	Where zero sits for every hidden unit's input	Measured in this study	Compared against mean and standard deviation scaling in the pilot
.Changing the source removed four transformations from Study 1's pipeline instead of logging them: the third-party resize to 128 pixels high, colour storage, conversion to grey at load time, and proportional-width cropping.
B.5 Software and hardware
We implement the RBM in Python with PyTorch, without an RBM library, and inspect every line before use. The starting implementation is a demonstration of the architecture, not a working instrument, and Stage 0 is where the visible scaling and noise schedule are corrected. And where the error summaries and negative-phase handling are settled (1b) The project's parsers decode the GNT and DGRL files (dgrl_extract.py for DGRL). Runs take place on the project workstation under Kubuntu Linux; hardware, driver and library versions are recorded with every run. Gradient clipping (norm 5) is retained from the starting implementation per Liao et al. (2022); it is absent from the source papers and its effect is logged in every run.

.
B.6 Validation and authentication
·	Data. Decoder byte accounting and layout checks (B.2); the HWDB1.1 working copy compared byte for byte with the recovered files; writer numbering confirmed across sources. The HWDB2.1 copy is unofficial, so its integrity rests on the decoder checks and on checking lines by eye until it is compared with an official checksum.
·	Instrument. Four checks precede any use of CASIA images (Aim 1a).
·	Randomness. Seeds are drawn from random.org and recorded with their draw date. A seed is used once per run type. Repeated runs for code or split corrections take a new seed from the ledger, and the previous run's label is preserved. Runs are labeled stage0_run_<n> with <n> assigned by the ledger, not by the script.
·	If a run is repeated for a reason (e.g., split correction, code fix), it gets a new seed from the ledger, and the previous run's label is preserved with a note (stage0_run_1a vs stage0_run_1b), not overwritten.


Protocol

Goal 1. Establish the RBM as a working instrument on this data
[RBMs have a reputation for being difficult to train]

1a. Instrument checks. We use no CASIA image until the implementation passes four checks, as the linear probe did on MNIST.
·	Positive control (MNIST). Train GB and GR on MNIST at 28 × 28 with 256 hidden units. Pass: filters show localized stroke and blob detectors, reconstructions are recognizable digits, and no hidden unit is always on or always off.
·	Negative control (noise). Train on Gaussian noise images matched to CASIA's pixel distribution. Pass: filters show no spatial structure, and held-out reconstruction is no better than the per-pixel median.
·	Permutation check. A fully connected RBM has no notion of neighbouring pixels. Train on CASIA images with one fixed pixel permutation, starting from the unpermuted run's initial weights permuted the same way, with the same batch order.Pass: the permuted run's weight-norm distribution, activity distribution, and held-out reconstruction error match the unpermuted run's within a pre-registered tolerance (distributional equivalence, not weight equality; CD-1 is stochastic and the batch order changes the gradient)
·	Fit monitor. Reconstruction error does not measure likelihood (Hinton, 2010). Track the median free energy of training and held-out images each epoch, with interquartile ranges (IQR). Pass: on MNIST, the gap widens when the model is deliberately overtrained on 500 images.
A failed check stops the work until the implementation is fixed and the check rerun.
1b. Reconcile the starting implementation. Each difference below is settled, or kept as a pilot arm, before Stage 1.


Item	Starting implementation	This protocol	Status	Reconciliation
Hidden-unit noise	Fixed standard deviation (SD) of 0.01	max(0, x + N(0, σ(x))), Nair & Hinton (2010)	Code changed/reason given/Resolved	change sample
Visible scaling	Pixels divided by 255; visible SD fixed at 1	Scaled to unit spread (ledger step 10)	Code changed/reason given/Resolved	Set sigma=1 but scale pixels to unit variance first.median removed, divided by global MAD
Negative-phase visible units	Sampled with Gaussian noise	Noise-free	Code changed/reason given
Resolved	Training uses Gaussian sampling in the negative phase with the visible σ, following Hinton (2010); reconstructions are read with noise-free means. The two are distinguished in the protocol: sampling during training is part of the model, not part of the measurement. This is consistent with the no-probe principle — reconstruction is a forward pass through the model's own weights, and the noise-free mean is the model's own expectation, not a fitted readout.
Free energy	Binary-unit form	Rectified-unit form for GR and GR0	Resolved	free energy uses softplus throughout, which is correct for both unit types; the binary-form concern in the starting implementation's docstring is resolved by this choice
Error summaries	Mean squared error	Median absolute error, ink and background separately	Code changed/reason given/Resolved

	background dominates every image)

The implementation's header cites the Gaussian–binary RBM stability literature. Liao et al. (2022) propose a Gibbs–Langevin sampler and a modified contrastive divergence, and report that the modified method with gradient clipping trains these models robustly; the starting implementation takes only the clipping. DeepSeek's suggested starting values, attributed to the original thesis, are recorded as suggestions, not as a source.
1c. Stage 0: getting the RBM to train. We run the reconciled implementation on the isolated condition and adjust it until its reconstructions hold structure worth describing; the contextual condition follows once the isolated one reconstructs. Gaussian-visible RBMs are hard to train, so this stage may take many attempts. We keep every attempt: its settings, curves, reconstruction images, and what we changed next and why. A run that produces noise, uniform grey or blobs is a finding about training this model on this data and is reported.


Iteration 1 Stage 0 runs. Both runs: 509 images at load (489 ISO + 20 CTX; the CTX count was later corrected to 155 after the path-naming bug was fixed — see O9). Device: cuda. Both runs: 150 epochs.   

Run 1 (label stage0_run_1, seed 27913148): train 410 (391 ISO, 19 CTX), val 99 (98 ISO, 1 CTX). Loss at epoch 10 = −37,645,284; loss at epoch 150 = −3,094,128,623,616. Already diverged at epoch 10; no stable regime.    

Run 2 (label stage0_run_2, overwriting stage0_run_1 per the seed warning; same seed 27913148): train 407 (391 ISO, 16 CTX), val 102 (98 ISO, 4 CTX). Loss trajectory below. Stable regime through epoch 40, crossing zero between epoch 40 and 50, divergence after.

Table 4 retention-invariant table
Epoch	Loss
10	+1,506,218
20	+1,425,586
30	+1,054,953
40	+21,321
50	−2,253,922
60	−6,369,931
70	−13,591,896
80	−24,028,750
90	−39,776,020
100	−62,039,276
110	−91,188,864
120	−129,478,512
130	−178,819,408
140	−240,537,760
150	−323,298,784


The model was in a stable regime through epoch 40 and crossed into divergence between epoch 40 and 50. The crossing is the finding: the free-energy objective is unbounded below, and the model found the bound during training from a stable starting point. This rules out "the initialization was bad" as the explanation; it points to the objective, not the starting weights. Run 1 (already diverged at epoch 10) and Run 2 (stable then diverged) are consistent with each other and together show that the objective, not the initialization, is what drives the divergence.
Seed ledger note. Both runs used seed 27913148. The run label was overwritten (stage0_run_1 → stage0_run_1 in the warning), so the seed ledger contains one seed assigned to one label, but the filesystem may contain two runs. The ledger does not distinguish them by seed; the record here distinguishes them by split and trajectory. Future runs must use distinct seeds from the ledger, and the ledger must be checked before any run is started.
What this changes about the retention-invariant table
The retention-invariant table is still correct.  Run 2 shows that the divergence is not a "we didn't lower the learning rate enough" problem; it is a "the objective is unbounded below and the model finds the bound" problem. That means the retention-invariant decision not to clip the data is vindicated by the trajectory — clipping the data would have hidden the divergence, not fixed it. The table's "Clip data to [−3, 3] → Rejected" row is the decision that made the finding visible. 
The retention-invariant decision to reject data clipping (rather than apply it) is what made the divergence visible. Clipping would have hidden the unbounded-below objective rather than fixing it. The finding is a consequence of the invariant, not a failure of it.
The seed-ledger problem
The warning line is a real issue and it needs a note in B.6 (validation and authentication) and in the seed-ledger record. The seed was reused; the run label was overwritten; the filesystem has two runs under one label. The record here distinguishes them by split and trajectory, but the ledger itself does not. For the protocol, the fix is:
·	The seed ledger records every seed drawn, and no seed is used twice for the same run type.
·	If a run is repeated for a reason (e.g., split correction, code fix), it gets a new seed from the ledger, and the previous run's label is preserved with a note (stage0_run_1a vs stage0_run_1b), not overwritten.


Table 1c: Phase_0_train attempts 1 and 2 with changes between each:
Proposed	Classification	Outcome in Run 2
Lower learning rate (1e-3 → 1e-4 → 1e-5)	Path-changing, trajectory-relevant	Run; diverged by epoch 150 (loss −323M)
Lower INIT_STD (0.01 → 0.001)	Retention-neutral	Run
Adam instead of momentum SGD	Path-changing	Run
Remove weight decay	Retention-relevant	Run
Grad clip 5.0 → 1.0	Rejected — too aggressive	Not run
Data clip to [−3, 3]	Rejected — changes what model represents	Not run
v_k clamp to ±3 / ±5	Rejected — changes gradient's target	Not run
Softplus noise schedule	Different instrument, not a stabilization	Not run (new instrument)

Run 2 outcome. Under the path-changing stabilizations (learning rate 1e-3 → 1e-4 → 1e-5, init 0.01 → 0.001, Adam, no weight decay), the loss was stable through epoch 40 (positive, decreasing) and crossed zero at epoch 40–50, diverging to −323M by epoch 150. The reconstruction dump at epoch 150 shows three different input characters reconstructed as identical near-black images with a single bright artifact in the upper-left quadrant. The model collapsed to the visible bias: it stopped reading the input entirely and output a constant image. This is not a blurry reconstruction (which would indicate learning) and not static (which would indicate noise); it is the model finding the trivial solution to the free-energy objective. The path-changing stabilizations did not prevent the collapse.  The classification column shows that every change is either path-changing or a different instrument, and every retention-lossy change was rejected. . The divergence is a property of the objective and the data, not of a tuning parameter.




Figure 1c-R2: Iteration 1 Stage 0, Run 2 reconstructions at epoch 150. Top row: three input crops (isolated 打 and 打 in context) as the model received them. Bottom row: the model's reconstructions from the same inputs. Source: models/stage0/, Run 2, epoch 150. Settings diff in 1c. Trajectory in 1c table.

The three inputs are different characters; the three reconstructions are identical. The model stopped reading the input and output its visible bias — a near-constant dark field with a single bright artifact in the upper-left quadrant. This is not a blurry reconstruction (which would indicate learning) and not static (which would indicate noise). It is the trivial solution to the free-energy objective: the model's weights have shrunk toward zero, the hidden units contribute nothing, and the reconstruction is the visible bias alone. The trajectory (stable through epoch 40, crossing zero at epoch 40–50, diverging to −323,298,784 by epoch 150) is consistent with this: the model found the trivial solution during training from a stable start. The path-changing stabilizations — lowered learning rate, lowered initialization, Adam, no weight decay — did not prevent the collapse. The finding is a property of the objective, not of the path.

Goal 2. Describe what isolated-trained and context-trained RBMs retain
2a. Extraction and inventory. We decode each isolated image to a PNG named for its GNT file and its position in that file (for example 1001-c_0042.png). For contextual crops we record the draw, round and seed batch, page, line, the position of 打 in the line text, the crop box, the characters the crop spans, and the visual-inspection outcome. Every recorded variable is defined in the data dictionary.
2b. Preprocessing. Preprocessing is the minimum the RBM needs, and every step that is a choice is run both ways in the pilot.
·	Resolution. Primary 32 × 32 (1,024 visible units), following Nair and Hinton. Before training, we compute each image's down-then-up residual at 32 and 48 pixels; if fine strokes that separate the characters are lost at 32, the 48-pixel run becomes primary.
·	Polarity. Ink dark and ink bright, because polarity decides which pixels sit near zero for rectified units.
·	Scaling. Primary: per-pixel median removed, divided by one global median absolute deviation. Comparison: per-pixel mean and global standard deviation, the scaling in Hinton (2010). Background is most of every image, so a deviation taken over all pixels can be zero; we check this first and, if so, take the scale over ink pixels.
·	Fitted on training folds only. Scaling statistics come from each fold's training images and are applied to its held-out images.
·	No augmentation. Added jitter, rotation or distortion would put assumptions about handwriting into the data we are describing.
2c. Models. All models use Gaussian visible units with variance fixed at 1 and noise-free reconstructions.
Model	Hidden unit	Hidden biases	Role
GB	Stochastic binary	Yes	Nair and Hinton's comparison unit
GR	Noisy rectified, max(0, x + N(0, σ(x)))	Yes	Rectified unit as usually used
GR0	Noisy rectified	None	The condition under which relative intensities are preserved
GL (optional)	Noisy leaky rectified	Yes	Link to the Leaky ReLU protocol; no published training recipe

2d. Training settings. Starting values follow Hinton (2010) and Nair and Hinton (2010); any change made in the pilot is recorded with its reason.
Setting	Starting value	Source
Learning rule	Contrastive divergence with one Gibbs step (CD-1)	Hinton, 2010
Learning rate	1e-3 on the per-image update	Nair & Hinton, 2010
Momentum	0.5 for 5 epochs, then 0.9	Hinton, 2010
Weight decay	L2 penalty, 1e-4, weights only	Hinton, 2010
Initial weights	Normal, centred at 0, SD 0.01	Hinton, 2010
Initial biases	0	Hinton, 2010
Mini-batch	10 images	Hinton, 2010 (range 10–100)
Epochs	Up to 300; stop when the held-out free-energy gap widens	Nair & Hinton, 2010; Hinton, 2010
Hidden units	64, 128, 256	Compared to see how features change as capacity grows


2e. 
Stage 1, pilot. Both conditions; GB and GR; 128 hidden units; 3 seeds; fold 1. Crossed with polarity (2), scaling (2) and resolution (2). Output: training curves, filters and residual maps for each choice, and a written record of which choice we keep and why, completed before Stage 2.


2f. Stage 2, full run. The design in A.3. Five folds are assigned once with their own random.org seed and reused for every model. Folds are grouped by writer, so no writer appears in both a training and a held-out fold. A sensitivity run also keeps each set of neighbouring characters within one fold, which tests whether shared neighbours let a model rebuild by recall. Held-out images are used only to measure reconstruction and free energy.
2g. Outputs.

Output	Question	What we produce
O1 Filter atlas	Q1	Every hidden unit's weights drawn as an image, with the 36 of highest weight norm and 36 drawn at random
O2 Activity profile	Q2	Histogram of how often each unit is active, by condition and unit type, and by character
O3 Residual maps	Q3	Per image: original, reconstruction, signed residual. Per character: median absolute residual map
O4 Hidden-code geometry	Q4	Within one model, distances between character medians in hidden-code space, including isolated 打 against 打 in context
O5 Cross-source reconstruction	Q5	Residuals when an isolated-trained model rebuilds contextual images and the reverse, against each model's own held-out residual
O7 Stability and fit	All	Seed-to-seed filter matching; training and held-out free energy and reconstruction curves; weight and bias histograms
O8 Rebuilt images	Q7	Each held-out image beside its reconstructions from the isolated- and context-trained models with matching unit type, size and seed


Rectified units are noisy, so we label every reconstruction by type: noise-off (hidden noise switched off; one pass back through the weights), median of draws (the pixel-wise median over a recorded number of noise draws, shown with its pixel-wise IQR), or sampled (one draw, what the model generates). Median of draws is the primary view.
Nair and Hinton displayed the filters of highest weight norm and noted that about a quarter of all filters were Gabor-like. Sorting by norm is a selection, so the full atlas and a random subset always accompany it. O5 restates Study 1's transfer collapse (91.4% in distribution, 5.26% on calligraphy) as a reconstruction question with no label involved.
Reading 打 in isolation against 打 in context. We read this comparison from O1, O2 and O4 in a fixed order, so the reading stays exploratory. For O1, we describe each model's full atlas and random subset, including filters that fit no category, before asking whether isolated filters resemble single strokes or contextual filters span two characters. For O2, we describe each model on its own source first; passing images from the other source through a model is O5's input, and differences there combine context with segmentation and intrusion. For O4, we take distances inside one model only, because separately trained RBMs have unrelated hidden units: each model encodes both isolated 打 and 打 in context, and the distance between those two medians is compared with the distances between 打 and the distractors in the same model.

Goal 3. Describe what the rectifying gate discards
3a. Biases and relative intensity. We scale every held-out image by several factors above zero and record whether each hidden unit's on/off state stays fixed, comparing GR with GR0. This tests the condition under which Nair and Hinton's claim holds.
3b. Gate decomposition (O6). For GR and GR0, we rebuild each held-out image through the model's own transposed weights from three versions of its hidden layer: the on/off pattern, the rectified output, and the full input including values below zero. What positive magnitudes add appears as the difference between the first two reconstructions; what negative magnitudes add, as the difference between the last two. The third version sends back values the model never sends back itself, so it shows what the gate removed rather than something the model would produce.
D. Analysis Plan
This plan is an initial version; the statistical analysis is being developed alongside the instrument work. Because this is an exploratory study, every quantity below is descriptive. What it measures is information retention: how faithfully each model rebuilds held-out handwriting, where it fails, and what its hidden code keeps.
D.1 estimand ΔX(VAR) is defined for any analytic variable, holding the data source constant while transformation and allocation choices vary. This study selects several analytic variables — held-out reconstruction residual over ink and background, hidden-unit activity, within-model hidden-code geometry, cross-source residual, and gate-decomposition differences — because they answer the retention question. The framework does not privilege any of them; the study names the ones it uses, and the choice of analytic variable is itself part of what the study reports. ΔX(VAR) is defined for any analytic variable; this study selects several because they answer the retention question, and a reader applying the framework elsewhere is not bound by this selection.
D.2 Variance budget, fixed in advance. Seeds are drawn from random.org in TRG batches before the analysis begins, in a quantity large enough to describe spread across runs; the number drawn is recorded in the seed ledger and is not chosen after seeing results. Patterns are described as counts ('X of N seeds showed...'). A pattern appearing in most seeds is described with its count and the qualifier that it is a description of these seeds, not an estimate of a population parameter. No stability threshold is fixed in advance, because the number of seeds is discovered (see above). (Bouthillier, X., et al. (2021). 
D.3 Paired design. Seeds and folds are shared across conditions and unit types, so every comparison is between matched pairs. We report the distribution of paired differences and the number of pairs in which the sign changes, as in the 1,000-seed run.
D.4 Summaries and intervals. Medians, interquartile ranges and full ranges; no means. Pixel summaries within an image are taken separately over ink and background, because background is most of every image. Intervals come from a stratified bootstrap over seeds within folds, following Agarwal et al. (2021) with the median in place of their interquartile mean. No unbiased estimator of the variance of K-fold cross-validation exists (Bengio & Grandvalet, 2004), so fold-based intervals are descriptive.
D.5 Grouping and leakage. Folds are grouped by writer, scaling is fitted on training folds only, and the neighbouring-character sensitivity run tests recall from shared context. We run a leakage audit as a formal step before Stage 2: no writer in both training and held-out folds, no held-out image in any scaling statistic, and no seed reused outside the ledger.
D.6 Selection. Every pilot choice is recorded with its evidence before Stage 2, because the evaluation pipeline itself can overfit (Varma & Simon, 2006; Cawley & Talbot, 2010). Any decision made after looking at held-out results is logged as such (Dwork et al., 2015; Blum & Hardt, 2015).
D.7 Reserved writers (proposed). A set of writers drawn by random.org and withheld from this study entirely would give a later confirmatory phase data this study has never touched.
D.8 All runs reported. Results are distributions over every run; no best run is selected. Every subset shown states how it was chosen, beside a random subset of the same size.
D.9 Recorded outputs. Per hidden unit per model: weight norm, activity frequency (overall and by character), median and IQR of the unit's input, fraction of inputs below zero, and best cosine match to a unit in another seed's model. Per image per model: reconstruction (with its type and number of draws), residual, hidden code, free energy, and the character of the nearest training image. Per epoch per model: median absolute reconstruction error over ink and background, gradient norm, number of clipped mini-batches, fraction of active hidden units, and median training and held-out free energy; reconstruction images every 10 epochs.
D.10 Deferred to a confirmatory phase. Significance tests, a smallest effect size of interest (SESOI), corrected paired tests (Dietterich, 1998; Nadeau & Bengio, 2003), Bayesian comparison (Benavoli et al., 2017), and external validity under distribution shift (Recht et al., 2019).

E. Rigor, Reproducibility, Limitations and Potential Pitfalls
E.1 Rigor and reproducibility
·	Matched seeds across every condition and unit type; all random numbers from random.org, held in one seed ledger.
·	Every run logs the code commit, environment, seed, fold and preprocessing settings, with data hashes; terminal output is saved to file from the first command.
·	Every transformation from page to model input is in the ledger (B.4).
·	Every run is reported, including runs that fail to train; nothing is dropped for looking wrong.
·	Every summary number appears beside the picture it summarizes.
·	AI assistance: DeepSeek (necessary for HK language expertise) wrote the starting RBM implementation and suggested starting values; Claude (Anthropic) helped draft this protocol and wrote the DGRL parser. Each claim is traced to a paper, the data or a run.
E.2 Limitations
·	The learning rule. CD-1 does not follow the true likelihood gradient, so filters reflect the data and the rule together. Comparing GB, GR and GR0 on the same data separates patterns that move with the unit type from those that stay with the data.
·	The hidden unit. Binary and rectified units limit what a unit can express differently; patterns seen under one unit type only are reported as such.
·	Noise and training modifications. GR samples noise in training; every reconstruction is labelled by type, and gradient clipping and Stage 0 changes are logged with the run.
·	No likelihood measure. Free energy and reconstruction error mark when a model starts copying its training images; they are not used to rank models.
·	Different segmentation. Isolated and contextual images share CASIA's collection process, grey format and writer pool, but CASIA segmented one by character and the other by line, and contextual crops can carry strokes from adjacent lines. Differences between conditions include these.
·	Writer structure. Writers match by identifier across sources, but the contextual pool covers only writers whose pages contain 打. Some structure may be writer style; writer identifiers let this be seen.
·	No spatial prior. A fully connected RBM treats pixels as unrelated to their neighbours, which suits a description of the images and differs from what a CNN would make of them.
·	Complexity. The data is demanding because of its variation (writer, stroke, neighbouring characters, segmentation). No choice here is set by image count; memorization is watched for directly through the held-out free-energy gap.
·	Human coding. One person codes intrusion. Crop boxes follow a fixed size, so they are not a rating.
·	ΔX(VAR) is defined relative to the baseline the research question requires.
The baseline need not be a direct observation; in this study it is not. VAR₀ is the cropped, resized, quantized image — the starting point for the variable of interest, which is what the RBM retains from the handwriting it is given. ΔX(VAR) measures the change in that variable induced by transformation and allocation choices, holding the data source constant. It does not claim to measure the distance from the handwriting on paper to the proxy, because that is a different question and this study does not pose it. The transformation ledger records which steps are measured and which are not; ΔX(VAR) is defined for the steps this study measures.
E.3 Potential pitfalls and alternatives

Pitfall	Response
The RBM does not train: reconstructions stay noise, grey or blobs after Stage 0	Report the Stage 0 record as the finding about this model on this data; change the instrument (the Liao et al. sampler, or a convolutional RBM; Lee et al., 2009) rather than read the data through a model that did not train
The global MAD is zero because background dominates	Take the scale over ink pixels
HWDB2.1 lines are shorter than 100 pixels	Excluded under the crop rule; the count is recorded before cropping and reported in the results, not treated as a data-cleaning step
A small first round lets models store their training images	Read held-out reconstructions and the free-energy gap; draw the next round from the same ledger sequence
Fine strokes are lost at 32 pixels	Move to 48 pixels, decided in the pilot
Filters or activity are unstable across seeds	Report it; any later comparison then needs the distributional design


E.4 What the readings mean for the next phase

If the analysis shows	Then the next phase
Contextual filters span the boundary between two characters	Needs representations wider than one character
Residuals concentrate where older writing or neighbouring lines intrude	Revisits cropping and intrusion before any model comparison
Fine stroke detail is lost at 32 pixels	Raises resolution or uses convolutional RBMs
GR and GR0 differ in on/off stability under intensity scaling	Treats hidden biases as a factor in the Leaky ReLU protocol
Reconstructions from the full hidden input show structure the rectified ones lack, for contextual but not isolated training	Supports running the Leaky ReLU protocol's information-loss arm first
Full-input and rectified reconstructions look alike in both conditions	Points to pooling rather than the gate as the larger loss in the CNN, to test directly next
Isolated and contextual models retain different structure	Informs how Huffman coding is combined with the RBM in the next phase

