# Progress and Results Log

Real-time chess piece detection with YOLOv8: a domain gap analysis.
AIML339, Anjola Sigbeku (300665808), Victoria University of Wellington.

Running record of experiments, findings, and decisions. Source material for
the evaluation and final reports.

---

## Status

| Stage | State |
|---|---|
| Dataset acquisition and conversion | Done |
| Data audit | Done, zero errors |
| YOLOv8 training on synthetic data | Done, 8 epochs |
| Stratified evaluation on synthetic test set | Done |
| Real-world image capture | Not started |
| Domain gap measurement | Not started |
| ResNet-50 baseline | Not started |
| Live webcam demo | Not started |

---

## Dataset

Rendered chess positions from Wölflein and Arandjelović, released openly at
[OSF](https://doi.org/10.17605/OSF.IO/XF3KA) under the chesscog project.
Chosen over the Roboflow alternatives, which held 46 to roughly 1,000 images
and in several cases did not distinguish piece colour at all. A label reading
only `bishop` cannot yield a FEN string.

| | Images | Boxes |
|---|---|---|
| train | 4,400 | 94,299 |
| val | 146 | 3,207 |
| test | 342 | 7,387 |
| **total** | **4,888** | **104,893** |

Twelve classes: six piece types across two colours, taken from FEN symbols
(uppercase white, lowercase black). Mean 21.5 boxes per image.

### Annotation quality

Every image ships a JSON containing the FEN, board corner coordinates, camera
angle, lighting mode, and exact per-piece bounding boxes computed in Blender by
projecting each piece's mesh vertices through the camera. Boxes are therefore
geometrically exact rather than hand-drawn.

Conversion to YOLO format produced **zero errors**: no malformed labels, no
out-of-range coordinates, no degenerate boxes, no unrecognised piece symbols.

### Validation check worth recording

`white-king` and `black-king` each total exactly **4,888** instances, matching
the image count precisely. Every legal chess position contains exactly one king
per side, so this independently confirms that FEN parsing, box extraction, and
image-to-annotation pairing are all correct. Any partial processing or symbol
misparse would break this identity.

The `min 2 / max 32` boxes-per-image bounds confirm it from the other side: 2 is
a bare-kings endgame, 32 a starting position. Both are exactly the legal limits.

### Class balance

Pawn-to-queen ratio is 8.6:1. This is intrinsic to chess, not a defect: eight
pawns per side against at most one queen. Kings appear in 100% of positions and
queens in about 64%, the signature of positions drawn from real games rather
than sampled uniformly.

**Decision: do not rebalance.** Report per-class AP so rare-class performance
stays visible instead of being absorbed into an aggregate dominated by pawns.

---

## Experiment 1: YOLOv8s on synthetic data

### Setup

| Parameter | Value |
|---|---|
| Model | yolov8s (11,140,228 parameters) |
| Epochs completed | 8 (of 50 requested) |
| Batch size | 16 |
| Image size | 640 |
| Seed | 42 |
| Hardware | Colab T4, 3.94 GB GPU memory used |
| Augmentation | HSV jitter, ±15° rotation, translate 0.1, scale 0.5, horizontal flip, mosaic |

Training was cut short when Colab reclaimed the session. Per-epoch checkpointing
meant `best.pt` survived on Drive. Results below come from that 8-epoch model.

Validation metrics had largely plateaued by then: mAP50-95 moved 0.907 → 0.910
→ 0.929 across epochs 3 to 5, climbing slowly from an already high base. The
synthetic task saturates quickly, which is itself relevant to the project's
argument.

### Results by render condition (test split, 342 images)

| Stratum | Images | mAP50-95 | mAP50 | Precision | Recall |
|---|---|---|---|---|---|
| **overall** | 342 | **0.9366** | 0.995 | 0.9975 | 0.9963 |
| angle 45–49° | 120 | 0.9302 | 0.995 | 0.9960 | 0.9944 |
| angle 50–54° | 118 | 0.9401 | 0.995 | 0.9967 | 0.9981 |
| angle 55–60° | 104 | 0.9425 | 0.995 | 0.9972 | 0.9983 |
| flash | 176 | 0.9413 | 0.995 | 0.9987 | 0.9989 |
| spotlights | 166 | 0.9329 | 0.995 | 0.9933 | 0.9964 |

### Per-class mAP50-95

| Class | mAP50-95 | | Class | mAP50-95 |
|---|---|---|---|---|
| black-knight | 0.9141 | | black-rook | 0.9399 |
| white-knight | 0.9188 | | black-queen | 0.9400 |
| black-pawn | 0.9257 | | white-king | 0.9429 |
| black-king | 0.9307 | | white-queen | 0.9473 |
| white-pawn | 0.9332 | | white-bishop | 0.9511 |
| black-bishop | 0.9366 | | white-rook | 0.9595 |

---

## Findings

### F1. Shallow camera angles are harder, monotonically

0.9302 → 0.9401 → 0.9425 as the camera rises from 45° to 60° above the board.

The renderer measures angle from horizontal, so 45° is side-on and 60° is closer
to overhead. Side-on views stack pieces in front of one another, so this is an
occlusion effect: the flatter the view, the more back-rank pieces are hidden
behind front-rank ones.

A monotonic trend across three independent bins is stronger evidence than any
single pairwise comparison would be.

### F2. Knights are the hardest class, and the reason is geometric

The two knights are the two worst classes (0.9141, 0.9188) against 0.9257 for
the next worst, a clear margin.

Every other piece is a surface of revolution: a pawn, rook, bishop, queen or
king looks the same from any direction. The renderer randomises each piece's
Z-rotation across a full 360°, so only the knight changes appearance as it
turns.

This matters for augmentation strategy. Rotation augmentation cannot fix a class
whose underlying object genuinely is not rotation-invariant; it can only expose
the model to more of that variation. The knight is the one class where extra
training data would plausibly help more than extra augmentation.

### F3. White pieces outperform black for every piece type

| Piece | White | Black | Difference |
|---|---|---|---|
| rook | 0.9595 | 0.9399 | +0.0196 |
| bishop | 0.9511 | 0.9366 | +0.0146 |
| king | 0.9429 | 0.9307 | +0.0122 |
| pawn | 0.9332 | 0.9257 | +0.0075 |
| queen | 0.9473 | 0.9400 | +0.0073 |
| knight | 0.9188 | 0.9141 | +0.0047 |

Six out of six in the same direction. Under a null of no effect, a two-tailed
sign test gives p ≈ 0.031. Mean advantage 0.011.

Dark pieces absorb light and expose less surface detail, so their edges localise
less precisely. That penalises mAP50-95, which is strict about box overlap,
while leaving mAP50 untouched.

Caveat: the six comparisons share the same images and so are not fully
independent, which weakens the nominal p-value.

### F4. mAP50 has no discriminative power on this task

It reads 0.995 in **every** stratum. Identical across camera angles, lighting
conditions, and overall.

All the signal lives in mAP50-95. Reporting mAP50 alone would have concealed
every finding above. Worth stating explicitly in the report as a methodological
point rather than leaving it implicit.

---

## Caveats

Strata contain 104 to 176 images and the differences are around 0.01. Taken
individually these are not clearly significant.

- **F1** holds up because the trend is monotonic across three bins.
- **F3** holds up because the direction is consistent across six pairs.
- **The lighting difference** (0.9413 flash vs 0.9329 spotlights) rests on a
  single comparison and should be reported as suggestive, not established.

The validation split is small at 146 images, roughly 90 queen instances, which
makes per-class model selection noisy. The paper's 90/3/7 split was kept anyway
to preserve comparability with Wölflein's published numbers.

Bounding boxes are **amodal**: computed from all mesh vertices including those
occluded by other pieces. A partly hidden piece receives a box covering its full
extent rather than only its visible part. Defensible for chess, but a real
difference from hand-annotated datasets and worth disclosing.

---

## Corrections needed to the design report

| Claim in design report | Actual |
|---|---|
| ~3,000 training images | 4,888 images, 104,893 instances |
| 13 classes | 12 (the text already described 12) |
| Roboflow + Kaggle as data sources | chesscog rendered dataset |
| yolov8m, 50 epochs | yolov8s, 8 epochs completed |

---

## Next steps

1. **Capture real chessboard photographs.** The domain gap cannot be measured
   without them. Vary lighting, angle, and chess set. This is the highest-value
   remaining work.
2. **Evaluate the synthetic-trained model on those real images.** The drop from
   0.9366 is the project's headline result.
3. **Fine-tune on a small number of real images** and re-measure, replicating
   Wölflein's 9.33 → 0.11 finding with an end-to-end detector rather than a
   three-stage pipeline.
4. ResNet-50 classification baseline for comparison.
5. Live webcam demo.
6. Optionally resume training to 50 epochs. Low priority: expected gain is
   roughly +0.02 mAP50-95, and it is not where the contribution lies.
