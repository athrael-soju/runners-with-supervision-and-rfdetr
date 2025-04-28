# Object Tracking System

An advanced object tracking system that uses the SORT (Simple Online and Realtime Tracking) algorithm to detect and track objects in videos. The system leverages RF-DETR for object detection and DeepSORT for tracking, providing robust object tracking capabilities with detailed reporting and visualization.

## Features

- **Object Detection**: Uses RF-DETR model to identify objects in video frames
- **Object Tracking**: Implements DeepSORT tracking algorithm for consistent tracking across frames
- **Reporting**: Generates detailed reports with tracking statistics
- **Visualization**: Creates insightful visualizations of tracking data
- **Real-time Processing**: Designed for efficient processing of video streams

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
```

## Project Structure

- `main.py`: Main script for object detection and tracking
- `report_generator.py`: Generates tracking reports and statistics
- `visualization_generator.py`: Creates visualizations from tracking data
- `input/`: Directory containing input videos
- `reports/`: Directory containing generated reports
- `visualizations/`: Directory containing generated visualizations

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