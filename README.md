# Chess Piece Detection (AIML339)

Chess piece detection with YOLOv8, with a ResNet-50 classifier as a
baseline. The research question is the domain gap: how much performance is lost
when a model trained on synthetic rendered images meets real smartphone photographs.

Anjola Sigbeku, 300665808, Victoria University of Wellington.

## Layout

```
chess-detection/
  src/
    config.py             paths, class schema, training defaults
    osf_download.py       resumable file downloader
    download_synth.py     fetch the chesscog rendered dataset from OSF
    download_chessred.py  fetch and inspect ChessReD real-photo dataset
    synth_to_yolo.py      convert chesscog annotations to YOLO format
    chessred_to_yolo.py   convert ChessReD annotations to YOLO format
    inspect_data.py       audit the data before training
    train_yolo.py         fine-tune YOLOv8
    finetune.py           fine-tune on N real photographs, sweep the curve
    evaluate.py           evaluate by camera angle and lighting condition
    make_crops.py         cut per-piece crops for the ResNet-50 baseline
    resnet_baseline.py    train and evaluate the ResNet-50 classifier
    compare_predictions.py  visualise side-by-side predictions
    plot_curve.py         plot the fine-tuning data efficiency curve
    demo.py               live webcam detection demo
  data/
    raw/               datasets exactly as downloaded
    interim/           after class remapping
    processed/         final YOLO-ready dataset
    crops/             per-piece crops for ResNet-50 baseline
  notebooks/           Colab training notebooks
  reports/             audit output
  runs/                training checkpoints and results
```

## Setup

```bash
pip install -r requirements.txt
```

No API keys are required for the primary data sources.

## Data sources

### 1. chesscog rendered dataset (primary)

4,888 Blender-rendered chess positions from Wölflein and Arandjelović, released
openly at [OSF](https://doi.org/10.17605/OSF.IO/XF3KA) with MIT-licensed code.

This is the largest and cleanest source available. Each image ships a JSON
carrying the FEN, board corner coordinates, camera angle, lighting mode, and
**exact per-piece bounding boxes** computed from Blender mesh geometry. Piece
symbols follow FEN convention, so the 12 colour-separated classes come for free.

```bash
pip install osfclient
python src/download_synth.py
python src/synth_to_yolo.py --src data/raw/render --dest data/processed/synth
python src/inspect_data.py --data data/processed/synth
```

About 2.5 GB. `download_synth.py` replaces chesscog's own downloader, which
pulls in torch, opencv and recap just to fetch files, clones into a temporary
directory, and cannot resume. A dropped connection there loses everything.

Here each file streams to a persistent `.part` file and resumes with an HTTP
Range request after an interruption, so **re-running the same command continues
where it stopped**. Splits already extracted are skipped without touching the
network. The script also joins the two-part `train.zip`/`train.z01` archive,
which Python's `zipfile` cannot read, and prints the first annotation so you
can confirm the per-piece `box` field exists before converting.

Add `--clean` to delete the archives after extraction and reclaim ~2.5 GB.

The converter also writes `render_metadata.csv`, which allows results to be
broken down by camera angle (45 to 60 degrees) and lighting mode (flash or
spotlights). This turns the domain gap from a single number into a curve.

If `pieces[].box` turns out to be absent from the released renders, boxes must
instead be derived from `corners` and `fen`, which is harder and approximate.
`download_synth.py` reports this at the end of its run.

### 2. ChessReD (real photographs, for the domain gap)

10,800 real smartphone photographs across 100 chess games, from Masouris and
van Gemert (VISAPP 2024), on
[4TU.ResearchData](https://doi.org/10.4121/99b5c721-280b-450b-b058-b2900b69a90f.v2)
under CC BY-NC-SA 4.0 (non-commercial).

Three handsets (iPhone 12, Huawei P40 Pro, Samsung Galaxy S8), angles from
top-down to oblique, natural and artificial light. A 2,078-image subset called
**ChessReD2K** additionally carries per-piece bounding boxes and board corners,
split 1,442 train / 330 val / 306 test.

That 306-image real test set is almost the same size as the 342-image synthetic
test set, which makes the synthetic-to-real comparison unusually clean. In the
test split each handset photographed exactly one game, so camera and game are
the same variable and no camera comparison is reported (see F8 in `RESULTS.md`).

```bash
python src/download_chessred.py                      # annotations, then inspect
python src/download_chessred.py --images --extract   # 4.68 GB
python src/chessred_to_yolo.py --src data/raw/chessred \
                               --dest data/processed/chessred2k
```

Two image archives are published. `chessred2k.zip` is **4.68 GB** and holds the
2,078 box-annotated images; `chessred.zip` is **24.57 GB** and adds 8,722 more
that carry no boxes and are unusable for detection. The default is the smaller
one. Pass `--full` only if you specifically need the remainder.

ChessReD orders its classes pawn, rook, knight, bishop, queen, king, whereas
this project uses pawn, knight, bishop, rook, queen, king. The converter maps
**by name**, and aborts if any name fails to match. Mapping by index would
relabel every rook as a knight while still producing plausible metrics.


## Workflow

```bash
# audit any dataset before training on it
python src/inspect_data.py --data data/processed/synth --json reports/audit.json
```

`inspect_data.py` exits with status 1 if it finds hard errors, so it can gate a
training run. It reports:

| Severity | Checks |
|---|---|
| Error | missing or orphaned label files, malformed lines, class ids out of range, coordinates outside [0,1], zero-area boxes |
| Warning | boxes below 0.05% of image area, images with more than 32 boxes, byte-identical duplicate images |
| Info | per-class instance counts per split, overall class imbalance ratio |

A high imbalance ratio is expected here. There are eight pawns per side but only
one king, so per-class precision and recall matter more than aggregate mAP.

## Training

The usual route is `notebooks/train_colab.ipynb`, which runs the whole pipeline
on a free Colab T4: clone, download, convert, audit, train, evaluate.

The notebook downloads the dataset fresh each session rather than storing it on
Drive. At about 7 GB extracted it would consume most of a free 15 GB Drive
quota, and Colab fetches it in a few minutes regardless. Only weights and
result tables are written to Drive.

Locally:

```bash
pip install ultralytics
python src/train_yolo.py
python src/evaluate.py --weights runs/<run>/weights/best.pt
```

Defaults to `yolov8s` for 50 epochs, which fits inside one free Colab session
on 4,400 images. The reported results come from an 8-epoch run, because Colab
reclaimed the session. The per-epoch checkpoint in `runs/best.pt` was kept.
`yolov8m` is roughly 3x slower and will usually be reclaimed partway through;
use `--resume` if you try it anyway. Checkpoints are written every epoch, so an
interruption costs one epoch rather than the whole run.

On Colab, mount Drive and write runs there so weights outlive the session:

```bash
python src/train_yolo.py --project /content/drive/MyDrive/AIML339/runs
```

Do not upload the dataset from your laptop. Run `download_synth.py` inside
Colab instead, where the connection is far faster.

`--no-aug` trains with every augmentation disabled, giving the ablation
baseline needed to quantify what augmentation actually contributes.

### Stratified evaluation

`evaluate.py` reports mAP separately for each camera angle band and lighting
mode, using the metadata captured during conversion. It builds a symlinked
subset per stratum and calls Ultralytics' own validator on each, so the numbers
are computed identically to the headline metric rather than reimplemented.

This is what turns "performance dropped on real images" into a statement about
*which conditions* break the model.

## Results at a glance

| Experiment | Result |
|---|---|
| YOLOv8s on rendered test images | 0.9366 mAP50-95 |
| Same model on 306 real photographs (zero-shot) | 0.189 mAP50-95, a 79.8% drop |
| Fine-tuned on 2 real photographs | 0.433 (mean of 3 seeds) |
| Fine-tuned on 25 / 50 real photographs | 0.741 / 0.826 (3 seeds each) |
| Fine-tuned on all 1,442 real photographs | 0.903 (one run) |
| ResNet-50 on ground-truth crops | 99.89% top-1 (upper bound on classification) |

Caveats: three seeds up to N = 50 and single runs above, 8 of 50 planned base
epochs, and no training time or inference latency measured. `RESULTS.md` has the
details and the file behind each number.

## Results log

Findings and experimental results live in `RESULTS.md`.

## Class balance is intrinsic

The 8.6x pawn-to-queen ratio is a property of chess, not a defect. There are
eight pawns per side and at most one queen, and queens get traded off. Kings
appear in exactly 100% of positions; queens in about 64%, which is the
signature of positions drawn from real games.

Do not rebalance it. Report per-class AP so rare-class performance is visible
instead of hidden behind an aggregate dominated by pawns.

## Class count

The project uses 12 classes: six piece types (pawn, knight, bishop, rook, queen, king)
across two colours (white, black). The chesscog and ChessReD datasets both confirmed
12 classes with zero errors on conversion. The design report incorrectly stated 13 classes.

## Status

- [x] Project scaffold and configuration
- [x] Dataset download and conversion: 4,888 synthetic images, 104,893 boxes, 0 errors
- [x] Dataset audit verified by king-count invariant
- [x] YOLOv8s training (8 epochs): 0.9366 mAP50-95 on synthetic test set
- [x] Stratified evaluation by camera angle and lighting condition
- [x] Domain gap measured: 0.1895 mAP50-95 on real photographs (79.8% drop)
- [x] Fine-tuning data efficiency curve (N=2 to 1442, seeds 42/43/44 for N≤50)
- [x] ChessReD2K per-piece crops extracted (43,149 crops across all splits)
- [x] ResNet-50 baseline trained: 99.89% top-1 accuracy on test crops
- [x] Live webcam demo script implemented (not evaluated)
- [x] Final report written
- [ ] Webcam demo tested against a live chessboard
- [ ] Own photographs for angle/lighting analysis with fixed chess set
