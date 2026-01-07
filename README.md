# CCTV Inference

Automated vehicle counting and cyclist demographic analysis from CCTV footage using computer vision.

> **Part of the [Cyclist Census](https://github.com/jcruz-ferreyra/cyclist_census) research project** - See the main repository for methodology, results, and the complete development pipeline.

<br>

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

<br>

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

<br>

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
poetry run python -m cctv_inference.process_cctv
```

**Output** (saved to `output_folder/video_name.parent`):
- `{video_name}_counts_part_{N}.json` - Count data with temporal breakdowns
- `{video_name}_counts_part_{N}.csv` - Same data in CSV format
- `{video_name}_output_part_{N}.mp4` - Annotated video (if `save_video: true`)
- `crops/{tracker_id}_frame{N}.jpg` - Cyclist images (if `keep_crops: true`)

<br>

## Structure

### Task Architecture

Each task within `cctv_inference` folder follows a consistent structure:

```
process_cctv/
├── __init__.py                 # Package initialization
├── __main__.py                 # Entry point - handles CLI and orchestration
├── config_min.yaml             # Minimum configuration reference
├── config_full.yaml            # Complete configuration reference
├── config.yaml                 # Processing configuration (user's working copy)
├── types.py                    # Context dataclass definition
├── cctv_processing.py          # Core processing logic (called from __main__.py)
└── *.py                        # Modular helper functions (called from cctv_processing.py)
```

**Context Pattern**:

All tasks use a context object to eliminate parameter passing complexity:

```python
@dataclass
class CCTVProcessingContext:
   # Configuration from YAML
   data_dir: Path
   models_dir: Path
   frame_processing: Dict[str, Any]
   detection: Dict[str, Any]
   ...
    
   # Runtime objects (initialized during setup)
   detection_model: Optional[Any] = None
   tracker: Optional[Any] = None
   ...
```

This pattern provides:
- Centralized configuration and state
- Automated path computation using `@property` decorators
- Initial validation using `__post_init__` method
- Intelligent defaults for missing optional configurations

<br>

## How it works

### Task 1: [Process CCTV Video](cctv_inference/process_cctv)

Runs the complete inference pipeline on a single video file, producing count data and optional annotated video output.

<details>
<summary><b>Details</b></summary>
<br>

**Processing Pipeline**:

1. Intialization
   - Initialize models and tracker
   - Initialize line counters
   - Initialize annotators (if video output enabled)

2. Processing Loop
   - Read frame from video
   - Run object detection (YOLO/RFDETR)
   - Update tracker with detections
   - Extract cyclists (person + bicycle IoU matching)
   - Classify cyclists (gender prediction)
   - Trigger line counters for each detection
   - Annotate frame (if video output enabled)

3. Partition Completion (every N minutes)
   - Aggregate gender classifications using temporal weighting
   - Calculate counts by vehicle type, gender, lane, and direction
   - Save results (JSON/CSV)
   - Save cyclist crops (if enabled)
   - Rotate video output file
   - Reset counters for next partition

4. Finalization
   - Process final partition
   - Clean up temporary files
   - Close video output

**Key algorithms**

- Cyclist extraction: IoU-based matching between person and bicycle detections
- Gender aggregation: Weighted mean of classifications for each track (near camera classifications weighted more)
- Line counting: Direction-aware crossing detection with double-count prevention using tracker ID tracking

</details>

<br>

## Additional Resources

For complete methodology, research context, and the full development pipeline, see the main **[Cyclist Census](https://github.com/jcruz-ferreyra/cyclist_census)** repository.

### Related Repositories

- **[detection_labelling](https://github.com/yourusername/detection_labelling)** - Dataset preparation for object detection models
- **[detection_training](https://github.com/yourusername/detection_training)** - YOLO/RFDETR model training pipeline
- **[classification_labelling](https://github.com/yourusername/classification_labelling)** - Dataset preparation for gender classification
- **[classification_training](https://github.com/yourusername/classification_training)** - CNN classifier training with Optuna optimization

### Support

For questions or issues:
- **GitHub Issues**: [cctv-inference/issues](https://github.com/jcruz-ferreyra/cctv-inference/issues)

### Citation

If you use this tool in your research, please cite:
```bibtex
@software{cyclist_census2025,
  title={Cyclist Census: Automated Demographic Analysis from CCTV},
  author={Ferreyra, Juan Cruz},
  institution={Universidad de Buenos Aires},
  year={2025},
  url={https://github.com/jcruz-ferreyra/cyclist_census}
}
```

### License

MIT License - see [LICENSE](LICENSE) file for details.
