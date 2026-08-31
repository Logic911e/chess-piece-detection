# Chess Piece Detection (AIML339)

Real-time chess piece detection with YOLOv8, with a ResNet-50 classifier as a
baseline. The research question is the domain gap: how much performance is lost
when a model trained on clean dataset images meets real webcam conditions.

Anjola Sigbeku, 300665808, Victoria University of Wellington.

## Layout

```
chess-detection/
  src/
    config.py          paths, class schema, training defaults
    download_data.py   fetch the Roboflow dataset
    osf_download.py    resumable OSF file downloader
    download_synth.py  fetch the chesscog rendered dataset from OSF
    synth_to_yolo.py   convert chesscog annotations to YOLO format
    inspect_data.py    audit the data before training
    train_yolo.py      fine-tune YOLOv8
    evaluate.py        evaluate by camera angle and lighting condition
  data/
    raw/               datasets exactly as downloaded
    interim/           after class remapping
    processed/         final YOLO-ready dataset
  notebooks/           Colab training notebooks
  reports/             audit output
  runs/                training checkpoints and results
```

## Setup

```bash
pip install -r requirements.txt
```

### Roboflow API key

The free **Public** plan is enough. No credit card is required, and a free key
can download public Universe datasets.

1. Sign up at [roboflow.com](https://roboflow.com)
2. Open your workspace, then **Settings -> API Keys**
3. Copy the key

Make it available to the scripts in one of two ways:

```bash
export ROBOFLOW_API_KEY="your_key_here"
```

or create a `.env` file in the repo root:

```
ROBOFLOW_API_KEY=your_key_here
```

Do not commit either. The key is private to your account.

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

### 2. Roboflow / real photographs (secondary)

Real photographs matter because the synthetic set is rendered. Only use
datasets whose classes distinguish colour: a label reading just `bishop`
cannot produce a FEN string and will corrupt the label space if merged.

```bash
python src/download_data.py
```

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
on a free Colab T4: clone, download, convert, audit, train, evaluate. Open it
directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Logic911e/chess-piece-detection/blob/main/notebooks/train_colab.ipynb)

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
on 4,400 images. `yolov8m` is roughly 3x slower and will usually be reclaimed
partway through; use `--resume` if you try it anyway. Checkpoints are written
every epoch, so an interruption costs one epoch rather than the whole run.

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

## Class balance is intrinsic

The 8.6x pawn-to-queen ratio is a property of chess, not a defect. There are
eight pawns per side and at most one queen, and queens get traded off. Kings
appear in exactly 100% of positions; queens in about 64%, which is the
signature of positions drawn from real games.

Do not rebalance it. Report per-class AP so rare-class performance is visible
instead of hidden behind an aggregate dominated by pawns.

## Note on class count

The design report states 13 classes while describing 12. Six piece types across
two colours is 12. Some public chess datasets add a thirteenth `board` class.
`inspect_data.py` reports what the downloaded data actually contains, so the
report should be corrected to match the real schema once the audit has run.

## Status

- [x] Project scaffold and configuration
- [x] Dataset download script
- [x] Dataset audit, verified against synthetic data with planted defects
- [x] chesscog render to YOLO converter, verified by exact round-trip
- [x] Dataset downloaded and converted: 4,888 images, 104,893 boxes, 0 errors
- [x] YOLOv8 training script with per-epoch checkpointing
- [x] Stratified evaluation by camera angle and lighting
- [ ] Run training and record baseline results
- [ ] ResNet-50 baseline
- [ ] Self-captured real-world test set
- [ ] Evaluation and domain gap analysis
- [ ] Live webcam demo
