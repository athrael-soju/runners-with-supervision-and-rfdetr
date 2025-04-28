# Object Tracking with SORT

This project uses SORT (Simple Online and Realtime Tracking) to track objects in videos. It leverages the RF-DETR model for object detection and DeepSORT for tracking.

## Features

- **Object Detection**: Detect objects in video frames using RF-DETR
- **Object Tracking**: Track detected objects across frames using DeepSORT
- **Video Generation**: Create annotated videos with bounding boxes and track IDs
- **Report Generation**: Generate detailed tracking reports with object metrics
- **Visualization**: Create data visualizations from tracking reports

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install supervision trackers rfdetr
   ```

3. For visualization support, install additional dependencies:
   ```
   pip install matplotlib seaborn opencv-python pandas
   ```

## Usage

### Basic Usage

To process a video with tracking and generate an output video:

```bash
python main.py
```

### Generate Reports

To process a video and generate detailed tracking reports:

```bash
python main.py --report
```

This will create two types of reports in the `reports` directory:
- A CSV summary report with basic tracking statistics
- A detailed JSON report with comprehensive tracking data

### Generate Visualizations

To generate data visualizations based on tracking results:

```bash
python main.py --report --visualize
```

This will create several visualization charts in the `reports/visualizations` directory:
- Object counts by class (bar chart)
- Tracking duration distribution (histogram)
- Object appearance timeline
- Detection density heatmap
- Confidence score distribution by class
- Object trajectory map

### Skip Video Generation

If you only want to generate reports/visualizations without creating an output video:

```bash
python main.py --report --visualize --no-video
```

## Report Details

The generated reports include the following information for each tracked object:

- **Track ID**: Unique identifier for each tracked object
- **Class**: The detected object class (from COCO classes)
- **First Seen**: Frame number when the object was first detected
- **Last Seen**: Frame number when the object was last detected
- **Duration**: How long the object was tracked (in frames and seconds)
- **Confidence**: Average confidence score for detections

The detailed JSON report includes additional metrics such as:
- Min/max confidence scores
- Formatted duration in hours:minutes:seconds
- Full video metadata

## Visualization Types

The following visualizations are generated:

1. **Objects by Class**: Bar chart showing count of unique objects detected for each class
2. **Duration Histogram**: Distribution of how long objects were tracked
3. **Object Timeline**: Timeline showing when each object appears in the video
4. **Detection Heatmap**: Heatmap showing where in the frame objects are most frequently detected
5. **Confidence Distribution**: Distribution of confidence scores by object class
6. **Trajectory Map**: Visualization of object movement paths through the video

## Customization

You can modify the following parameters in the code:
- `SOURCE_VIDEO_PATH`: Path to the input video
- `CONFIDENCE_THRESHOLD`: Minimum confidence score for detections (default: 0.5) 