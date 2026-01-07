# CCTV Inference

Automated vehicle counting and cyclist demographic analysis from CCTV footage using computer vision.

> **Part of the [Cyclist Census](https://github.com/yourusername/cyclist_census) research project** - See the main repository for methodology, results, and the complete development pipeline.

## Overview

Offline CCTV video processing pipeline that detects, tracks, and counts vehicles while capturing cyclist gender demographics and bike lane compliance data. Designed for urban transportation research and planning.

### Capabilities

- **Multi-object detection and tracking**: YOLO/RFDETR detection + ByteTrack multi-object tracking
- **Cyclist identification**: Person-bicycle matching using IoU-based association
- **Gender classification**: EfficientNet/ResNet models with temporal aggregation for robust predictions
- **Directional counting**: Line-based counters with configurable class filtering and direction tracking
- **Temporal aggregation**: Configurable time intervals (e.g., 15-minute partitions) for count summaries
- **Google Colab support**: Checkpoint-based processing with automated resume on disconnection

### Output

- **Count data** - JSON/CSV files with temporal breakdowns by vehicle type, gender, lane compliance, and direction
- **Annotated videos (Optional)** - Bounding boxes, tracker IDs, classification labels, and live count overlays
- **Cyclist crops (Optional)** - Individual images per tracked cyclist for verification and quality control

## Installation

### Prerequisites
   - Python 3.11+
   - Poetry (for dependency management)
   - GPU recommended (CUDA-compatible)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/jcruz-ferreyra/cctv-inference.git
   cd cctv-inference
   ```

2. **Install dependencies**
   ```bash
   poetry install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your paths:
   # DATA_DIR=/path/to/your/main/data/folder
   # MODELS_DIR=/path/to/your/main/models/folder
   ```

4. **Download model weights**
   
   Place trained model weights in your `MODELS_DIR`:
   ```
   models/
   ├── path/to/detection/model/weights.pt
   └── path/to/classification/model/weights.pt
   ```

5. **Prepare your data**
   ```
   data/
   └── path/to/your/video/folder
      ├── your_video.avi
      └── line_counters.json
   ```

## Quick Start

### Task 1: [Process CCTV Video](cctv_inference/process_cctv)
Runs the complete pipeline: detection → tracking → classification → counting.

**Configuration**:

This task uses a two-layer configuration system:

1. Processing Configuration ([`config.yaml`](cctv_inference/process_cctv/config.yaml))

   YAML file defining paths, processing parameters, model settings, and output options:
   
   ```yaml
   # Input/Output
   input_folder: path/to/your/video/folder   # Directory containing input video
   output_folder: path/to/output/results     # Directory for output results
   video_name: your_video.avi                # Video filename (relative to input_folder)
   
   # Line counters
   line_counters:
     lines_file: "line_counters.json"        # Line configuration file (relative to input_folder)
   
   # Frame processing
   frame_processing:
     inference_interval: 5                   # Process every Nth frame
     partition_minutes: 15                   # Time interval for result aggregation
     video_start_time: "2024-08-01 17:30:00" # Video start timestamp
   
   # Detection
   detection:
     model_architecture: "yolo"              # "yolo" or "rfdetr"
     model_weights: "path/to/detection/weights.pt"
     class_label:                            # Class ID to label mapping
       0: person
       1: car
       2: bicycle
       3: motorcycle
   
   # Classification
   classification:
     enabled: true
     model_architecture: "efficientnet_b0"
     model_weights: "path/to/classification/weights.pt"
      labels: [female, male]
   
   # Output
   output:
     save_video: false                       # Generate annotated video
     keep_crops: false                       # Save individual cyclist images
   ```
   
   For a complete reference with all available options and detailed comments, see [`config_full.yaml`](cctv_inference/process_cctv/config_full.yaml).

2. Line Counter Configuration (`line_counters.json`)

   JSON file defining counting lines, placed in the `input_folder` directory. Each line specifies vehicle detection and counting behavior using [supervision LineZone](https://supervision.roboflow.com/latest/detection/tools/line_zone/) parameters:
   
   ```json
   [
     {
       "count_id": "StreetName_N-S_lane",
       "lane": true,
       "direction": "out",
       "anchor": "top",
       "labels": ["bicycle", "motorcycle"],
       "coords": [[376, 1037], [629, 1121]]
     },
     {
       "count_id": "StreetName_N-S_car",
       "lane": false,
       "direction": "out",
       "anchor": "center",
       "labels": ["car", "bus", "truck"],
       "coords": [[380, 1040], [632, 1124]]
     }
   ]
   ```
   
   Field descriptions:
   - `count_id` - Unique identifier for this counting line (format: `{location}_{direction}`.) 
   - `lane` - `true` for bike lane counters, `false` for car lanes
   - `direction` - Count direction: `"in"` or `"out"` (see [LineZone docs](https://supervision.roboflow.com/latest/detection/tools/line_zone/))
   - `anchor` - Detection anchor point: `"center"`, `"top"`, or `"bottom"` (see [LineZone docs](https://supervision.roboflow.com/latest/detection/tools/line_zone/))
   - `labels` - Vehicle classes to count (must match `class_label` values in `config.yaml`)
   - `coords` - Line coordinates as `[[x1, y1], [x2, y2]]` (see [LineZone docs](https://supervision.roboflow.com/latest/detection/tools/line_zone/))


**Run**:
```bash
poetry run python -m process_cctv
```

**Output** (saved to `output_folder/video_name.parent`):
- `{video_name}_counts_part_{N}.json` - Count data with temporal breakdowns
- `{video_name}_counts_part_{N}.csv` - Same data in CSV format
- `{video_name}_output_part_{N}.mp4` - Annotated video (if `save_video: true`)
- `crops/{tracker_id}_frame{N}.jpg` - Cyclist images (if `keep_crops: true`)

## How It Works

### Architecture
File structure + context pattern

### Data Organization
Where to put inputs/outputs

### Processing Pipeline
Diagram + flow description

### Key Components
Algorithm explanations

## Output Format
(if applicable)

## Troubleshooting
(if needed)

## Additional Resources
Link to main cyclist_census repo


============================================================================================
PREVIOUS
============================================================================================
# Cyclist Census

Automated cyclist counting and demographic analysis from CCTV footage using computer vision.

## Overview

Real-time CCTV video processing pipeline that detects, tracks, and counts cyclists while capturing gender demographics and bike lane compliance data. Designed for urban transportation research and planning.

### Key Features

- **Multi-object detection and tracking** - YOLO/RFDETR + ByteTrack
- **Cyclist identification** - Person-bicycle matching with IoU-based association
- **Gender classification** - EfficientNet/ResNet models with temporal aggregation
- **Bike lane compliance** - Polygon-based zone analysis
- **Directional counting** - Line-based counters with configurable filtering
- **Temporal aggregation** - Configurable intervals (e.g., 15-minute partitions)
- **Google Colab support** - Checkpoint-based processing with resume capability

### Output

- Annotated videos with bounding boxes, labels, and live count overlays
- JSON/CSV count results with temporal metadata
- Cyclist crops for verification and quality control
- Per-partition checkpointing for interrupted processing

---

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry (for dependency management)
- GPU recommended (CUDA-compatible)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/jcruz-ferreyra/cyclist_census.git
   cd cyclist_census
   ```

2. **Install dependencies**
   ```bash
   poetry install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your paths:
   # DATA_DIR=/path/to/your/main/data/folder
   # MODELS_DIR=/path/to/your/main/models/folder
   ```

4. **Download model weights**
   
   Place trained model weights in your `MODELS_DIR`:
   ```
   models/
   ├── detection/
   │   └── yolov8m_v1/
   │       └── weights/
   │           └── best.pt
   └── classification/
       └── efficientnet_b0_v1/
           └── weights/
               └── best.pt
   ```

5. **Prepare your data**
   ```
   data/
    └── cctv_videos/
        └── your_video/
            ├── your_video.avi
            ├── bike_lanes.json
            └── line_counter.json
   ```

### Basic Usage

**Local execution:**
```bash
poetry run python -m cyclist_census.process_cctv
```

**Google Colab:**

Use the provided notebook: `notebooks/colab_inference.ipynb`

The notebook handles:
- Google Drive mounting
- Automatic dependency installation
- Video transfer to local Colab storage
- Checkpoint-based processing with resume support
- Output save to mounted Google Drive

---

## Configuration

The pipeline uses a two-layer configuration system:

### 1. Environment Variables (`.env`)

```bash
DATA_DIR=/path/to/data          # Local data directory
MODELS_DIR=/path/to/models      # Model weights directory
```

### 2. YAML Configuration (`config.yaml`)

**Minimal configuration example:**

```yaml
# Input/Output
input_folder: data/raw/cctv_videos
output_folder: data/results
video_name: sample_video.avi

# Frame processing
frame_processing:
  inference_interval: 5           # Process every 5th frame
  partition_minutes: 15           # 15-minute result intervals
  video_start_time: "2024-08-01 17:30:00"

# Detection
detection:
  model_architecture: "yolo"
  model_weights: "detection/yolov8m_v3_1/weights/best.pt"
  class_label:
    0: person
    1: car
    2: bicycle
    3: motorcycle

# Tracking
tracking:
  tracker_type: "bytetrack"

# Classification
classification:
  enabled: true
  model_architecture: "efficientnet_b0"
  model_weights: "classification/efficientnet_b0_gender/weights/best.pt"

# Spatial configuration
bike_lanes:
  polygon_file: "bike_lanes.json"

line_counters:
  lines_file: "line_counter.json"

# Output
output:
  save_video: true
  save_counts: true
  formats: [json, csv]
```

For a complete list of all available configuration options with detailed comments and default values, see:

**[`cyclist_census/process_cctv/config_full.yaml`](cyclist_census/process_cctv/config_full.yaml)**

### Spatial Configuration

**Line Counters** (`line_counter.json`):

```json
[
    {
        "count_id": "OvidioLagos_S-N",
        "lane": true,
        "direction": "out",
        "anchor": "top",
        "labels": ["bicycle", "motorcycle"],
        "coords": [[376, 1037], [629, 1121]]
    },
    {
        "count_id": "OvidioLagos_S-N",
        "lane": false,
        "direction": "out",
        "anchor": "top",
        "labels": ["bicycle", "motorcycle"],
        "coords": [[0, 903], [376, 1037]]
    },
    ...
]
```

```yaml
line_counters:
  source: "F053_LAG_H35/240729/line_counters.json"
```

Fields:
- `count_id` - Unique identifier
- `lane` - Bike lane compliance tracking flag
- `direction` - "in", "out", or "both"
- `anchor` - Detection point to check
- `labels` - Vehicle class to count
- `coords` - Line start/end `[[x1, y1], [x2, y2]]`

---


## How It Works

### Architecture

The pipeline follows a context-based design pattern where all configuration, paths, and runtime objects are organized in a `CCTVProcessingContext` dataclass:

```python
@dataclass
class CCTVProcessingContext:
    # Environment paths
    data_dir: Path
    models_dir: Path
    
    # Configuration sections
    frame_processing: Dict[str, Any]
    detection: Dict[str, Any]
    tracking: Dict[str, Any]
    classification: Dict[str, Any]
    
    # Runtime objects (initialized during setup)
    detection_model: Optional[Any] = None
    classification_model: Optional[Any] = None
    tracker: Optional[Any] = None
```

### Processing Pipeline

```
┌─────────────────────────────────────────────────────────┐
│ 1. Setup & Initialization                               │
│    - Load models (YOLO/RFDETR, EfficientNet/ResNet)     │
│    - Initialize tracker (ByteTrack)                     │
│    - Load spatial configs (bike lanes, count lines)     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Frame-by-Frame Processing                            │
│    For each frame:                                      │
│    a. Run detection (YOLO/RFDETR)                       │
│    b. Update tracker (ByteTrack)                        │
│    c. Extract cyclists (person+bicycle IoU matching)    │
│    d. Classify cyclists (gender, if enabled)            │
│    e. Trigger line counters                             │
│    f. Check bike lane compliance                        │
│    g. Annotate frame (if video output enabled)          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Partition Completion                                 │
│    At each partition boundary:                          │
│    - Aggregate classifications (temporal weighting)     │
│    - Calculate counts by gender/vehicle/lane            │
│    - Save results (JSON/CSV)                            │
│    - Rotate video output file                           │
│    - Checkpoint progress (Colab)                        │
│    - Reset counters                                     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Finalization                                         │
│    - Process final partition                            │
│    - Cleanup crops (if keep_crops=false)                │
│    - Sync to Drive (Colab)                              │
└─────────────────────────────────────────────────────────┘
```

### Key Components

#### Cyclist Extraction

Matches person and bicycle detections using IoU (Intersection over Union):

```python
# 1. Calculate IoU between all bicycles and persons
iou_matrix = box_iou_batch(bicycle_boxes, person_boxes)

# 2. Find best matching person for each bicycle
best_person_idx = argmax(iou_matrix, axis=1)
max_iou = max(iou_matrix, axis=1)

# 3. Filter by minimum threshold (0.1)
valid_matches = max_iou > 0.1

# 4. Extract person crop with padding
crop = frame[y1-20:y2+10, x1-10:x2+10]  # Extra padding for head
```

#### Gender Classification

Multiple observations per cyclist track are aggregated using temporal weighting (higher weight for crops closer to the camera):

```python
# Direction-aware weighting
if direction == "in":
    weights = [n, n-1, ..., 2, 1]  # Earlier frames weighted more
elif direction == "out":
    weights = [1, 2, ..., n-1, n]  # Later frames weighted more

# Weighted voting
for prediction, weight in zip(predictions, weights):
    votes[prediction] += weight

final_gender = max(votes, key=votes.get)
```

#### Line Counting

Directional counting with double-crossing prevention:

```python
# Filter detections by line-specific classes
line_detections = detections[class_mask]

# Remove already-crossed trackers
line_detections = line_detections[~crossed_mask]

# Trigger crossing detection
crossed_in, crossed_out = line_zone.trigger(line_detections)

# Track crossed IDs to prevent double counting
crossed_tracker_ids.update(newly_crossed)
```

---

## Output Format

### Count Results

**JSON format** (`{video_name}_counts_part_{partition_id}.json`):

```json
[
  {
    "count_id": "OvidioLagos_S-N",
    "vehicle": "bicycle",
    "lane": true,
    "gender": "male",
    "count": 9,
    "start_time": "2024-07-29T13:21:24",
    "end_time": "2024-07-29T13:30:01.080000"
  },
  {
    "count_id": "OvidioLagos_S-N",
    "vehicle": "motorcycle",
    "lane": false,
    "gender": null,
    "count": 8,
    "start_time": "2024-07-29T13:21:24",
    "end_time": "2024-07-29T13:30:01.080000"
  },
  {
    "count_id": "OvidioLagos_N-S",
    "vehicle": "bicycle",
    "lane": true,
    "gender": "female",
    "count": 3,
    "start_time": "2024-07-29T13:21:24",
    "end_time": "2024-07-29T13:30:01.080000"
  },
  ...
]
```

**CSV format** - Same data in tabular form for easy analysis.

### Video Output

Annotated videos with:
- Bounding boxes (color-coded by class)
- Tracker IDs
- Classification labels (gender for cyclists)
- Live count overlay

**Partitioned output** (when `save_single: false`):
```
F053_LAG_H35_240729_22_output_part_0001.mp4
F053_LAG_H35_240729_22_output_part_0002.mp4
...
```

### Cyclist Crops

Individual cyclist images for verification:
```
{tracker_id}_frame{frame_number}.jpg

Examples:
42_frame001523.jpg
58_frame002891.jpg
```

---

## Google Colab Usage

The pipeline is optimized for Google Colab with checkpoint-based processing and automatic resume support.

### Setup

1. **Open the notebook**
   - Upload `notebooks/colab_inference.ipynb` to your Drive
   - Or use: [Open in Colab](link)

2. **Mount Google Drive**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Configure paths in notebook**
   ```python
   DRIVE_DATA_DIR = "/content/drive/MyDrive/cyclist_census/data"
   DRIVE_MODELS_DIR = "/content/drive/MyDrive/cyclist_census/models"
   ```

4. **Run processing**
   - Video is copied to local Colab storage for faster processing
   - Results are saved to Drive after each partition
   - Session can disconnect and resume automatically

### Checkpoint & Resume

**How it works:**

1. Video is processed in partitions (e.g., 15-minute intervals)
2. After each partition:
   - Count results saved to Drive
   - Cyclist crops saved to Drive
   - Progress checkpointed
3. On disconnect:
   - Last completed partition is detected
   - Incomplete crops are cleaned up
   - Processing resumes from next partition

**Manual resume:**

If you need to manually resume from a specific partition:

```yaml
frame_processing:
  start_from_partition: 5  # Resume from 6th partition (0-indexed)
```

The system will:
- Calculate correct starting frame
- Skip already-processed partitions
- Continue from specified partition

### Performance Tips

- **Use GPU runtime** - 10-20x faster than CPU
- **Adjust inference_interval** - Higher values (10-15) for faster processing
- **Partition size** - 15-30 minutes balances checkpoint frequency and overhead
- **Video output** - Disable (`save_video: false`) if only counts are needed

---

## Supported Models

### Detection Models

- **YOLO** (via ultralytics) - YOLOv8, YOLOv11, etc.

### Classification Models

- **EfficientNet** - B0, B3 (via timm)
- **ResNet** - 50, 101 (via timm)

Custom models can be integrated by implementing architecture-specific loading functions.

---

## Development Pipeline

This inference system uses custom-trained detection and classification models. The complete development pipeline consists of:

### Dataset Preparation

- **[detection_labelling](link)** - CCTV frame extraction, BYOL-based sampling, SIFT deduplication, format conversion
- **[classification_labelling](link)** - Cyclist crop extraction and organization

### Model Training

- **[detection_training](link)** - YOLO/RFDETR training with MLflow tracking
- **[classification_training](link)** - CNN training with Optuna hyperparameter optimization

Key innovations:
- Threshold optimization methodology (custom per-model thresholds)
- Class-specific data augmentation
- Layer freezing analysis for transfer learning

---

## Troubleshooting

### Common Issues

**Issue: CUDA out of memory**

Solution:
```yaml
frame_processing:
  frame_batch_size: 16  # Reduce from 32
  inference_interval: 10  # Process fewer frames
```

**Issue: Colab session disconnects**

Solution: The checkpoint system handles this automatically. Simply re-run the notebook - processing will resume from the last completed partition.

**Issue: No cyclists detected**

Possible causes:
- Detection confidence too high - lower `category_confidence.bicycle`
- IoU threshold for cyclist matching too high (fixed at 0.1)
- Check video quality and camera angle

**Issue: Gender classification seems random**

Possible causes:
- Wrong model threshold - use recommended values per architecture
- Insufficient crops per track for temporal aggregation
- Low-quality crops (poor lighting, occlusion)

**Issue: Double counting at lines**

Check:
- Line placement and direction configuration
- `anchor` setting matches detection point you want to count
- Track buffer (`track_buffer`) not too long for your scene

### Logging

Debug logging is saved to `DATA_DIR/logs/process_cctv.log`:

```bash
tail -f /path/to/data/logs/process_cctv.log
```

Log levels:
- `INFO` - Major pipeline steps
- `DEBUG` - Detailed processing information
- `ERROR` - Failures and exceptions

---

## Research Context

**Institution:** Northeastern University, Boston Area Research Initiative (BARI)

**Research Focus:** Urban informatics, environmental monitoring, smart city applications

**Use Case:** Automated cyclist census from CCTV footage for transportation planning and policy development.

---

## Project Structure

```
cyclist_census/
├── cyclist_census/
│   ├── process_cctv/
│   │   ├── __main__.py              # Entry point
│   │   ├── cctv_processing.py       # Main pipeline
│   │   ├── types.py                 # Context dataclass
│   │   ├── frame_counter.py         # Frame iteration & partitioning
│   │   ├── classifications.py       # Classification container
│   │   └── sink_manager.py          # Video output management
│   └── utils/
│       ├── config.py                # Configuration loading
│       └── logging.py               # Logging setup
├── docs/                            # Documentation
├── models/                          # Model weights (not in repo)
├── notebooks/
│   └── colab_inference.ipynb        # Colab notebook
├── config.yaml                      # Pipeline configuration
├── .env.example                     # Environment template
├── pyproject.toml                   # Poetry dependencies
└── README.md                        # This file
```

<!-- ---

## Citation

If you use this system in your research, please cite:

```bibtex
@software{cyclist_census2024,
  title={Cyclist Census: Automated Demographic Analysis from CCTV},
  author={Your Name},
  institution={Northeastern University, Boston Area Research Initiative},
  year={2024},
  url={https://github.com/jcruz-ferreyra/cyclist_census}
}
```

--- -->

<!-- ## License

[Your chosen license]

---

## Contact

For questions or issues:
- GitHub Issues: [link]
- Email: [your.email@university.edu] -->
