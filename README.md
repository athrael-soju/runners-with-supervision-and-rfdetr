# Object Tracking System

An advanced object tracking system that uses the SORT (Simple Online and Realtime Tracking) algorithm to detect and track objects in videos. The system leverages RF-DETR for object detection and DeepSORT for tracking, providing robust object tracking capabilities with detailed reporting and visualization.

## Features

- **Object Detection**: Uses RF-DETR model to identify objects in video frames
- **Object Tracking**: Implements DeepSORT tracking algorithm for consistent tracking across frames
- **Reporting**: Generates detailed reports with tracking statistics
- **Visualization**: Creates insightful visualizations of tracking data
- **Real-time Processing**: Designed for efficient processing of video streams
- **Flexible Video Processing**: Control video duration, FPS, and starting position
- **Modular Architecture**: Clean separation of processing logic for easier extensibility

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- Pandas

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/trackers.git
cd trackers

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -U git+https://github.com/roboflow/rf-detr.git
pip install -U git+https://github.com/roboflow/trackers.git
pip install -U git+https://github.com/roboflow/supervision.git
pip install matplotlib seaborn pandas
```

## Usage

### Basic Usage

```bash
python main.py
```

### Advanced Usage

```bash
# Generate tracking reports
python main.py --report

# Generate visualizations
python main.py --visualize

# Generate both reports and visualizations
python main.py --report --visualize

# Control video processing
python main.py --seconds 30 --fps 15 --start_frame 900
```

### Video Processing Parameters

The system provides options to customize video processing:

- `--seconds`: Process only a specific duration of the video (in seconds)
- `--fps`: Set the output frame rate (frames per second)
- `--start_frame`: Begin processing from a specific frame number
- `--low_fps`: Optimize tracker for low frame rate videos (increases tracking persistence and reduces motion dependency)

Examples:

```bash
# Process only the first 10 seconds
python main.py --seconds 10

# Process at reduced frame rate (3 FPS)
python main.py --fps 3

# Process at reduced frame rate with low-FPS tracking optimization
python main.py --fps 3 --low_fps

# Process 30 seconds starting from frame 900
python main.py --seconds 30 --start_frame 900

# Process 15 seconds starting from frame 1500 at 10 FPS
python main.py --seconds 15 --start_frame 1500 --fps 10
```

## Project Structure

- `main.py`: Entry point script that handles command-line arguments and initialization
- `processor.py`: Contains the VideoProcessor class that handles video processing logic
- `report_generator.py`: Generates tracking reports and statistics
- `visualization_generator.py`: Creates visualizations from tracking data
- `input/`: Directory containing input videos
- `reports/`: Directory containing generated reports
- `visualizations/`: Directory containing generated visualizations

## Class Architecture

### VideoProcessor

The `VideoProcessor` class in `processor.py` handles all video processing tasks:

```python
processor = VideoProcessor(
    source_path="input/video.mp4",  # Input video path
    model=model,                    # Detection model
    tracker=tracker,                # Object tracker
    confidence_threshold=0.5,       # Detection confidence threshold
    reporter=reporter               # Optional tracking reporter
)

# Process with custom parameters
processor.process_video(
    seconds=30,                     # Duration to process
    output_fps=15,                  # Output frame rate
    start_frame=900                 # Starting frame
)
```

## Reports

The system generates two types of reports:
- **Summary CSV**: Basic statistics on each tracked object
- **Detailed JSON**: Comprehensive tracking data for further analysis

## Visualizations

Four types of visualizations are generated:
- **Objects by Class**: Bar chart showing object counts by class
- **Tracking Duration Histogram**: Distribution of how long objects are tracked
- **Object Timeline**: Timeline showing when objects appear in the video
- **Confidence Distribution**: Distribution of confidence scores by class

## Examples

After processing a video, you can find:
- The processed video with tracking annotations in the `output/` directory
- Reports in the `reports/` directory
- Visualizations in the `visualizations/` directory

## License

[MIT License](LICENSE) 