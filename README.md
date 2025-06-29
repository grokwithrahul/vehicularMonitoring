# Vehicle Tracking and License Plate Recognition System

A robust Python-based system for detecting, tracking, and analyzing vehicles in video footage using YOLOv11 and ByteTrack. The system automatically identifies unique vehicles, captures their best frames, and performs license plate recognition with optional AI-powered vehicle analysis.

## Features

- **Real-time Vehicle Detection**: Uses YOLOv11 for accurate vehicle detection
- **Robust Tracking**: ByteTrack algorithm handles occlusions and ID switching
- **Smart Vehicle Filtering**: Saves only unique vehicles based on movement patterns and size
- **License Plate Recognition**: Automatic detection and extraction of license plates
- **AI-Powered Analysis**: Optional OpenAI integration for vehicle details (make, model, color, etc.)
- **Adaptive Processing**: Frame skipping for performance optimization
- **Comprehensive Output**: Generates both individual frames and review clips

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for acceleration)
- OpenAI API key (optional, for vehicle analysis)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd car_recognition
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required models** (if not already included):
   - `yolo11m.pt` - YOLOv11 medium model for vehicle detection
   - `license_plate_detector.pt` - License plate detection model

## Usage

### Basic Usage

Run the vehicle tracker on a video file:

```bash
python object_tracker.py path/to/your/video.mp4
```

### Advanced Usage

```bash
python object_tracker.py path/to/your/video.mp4 \
    --model-path yolo11m.pt \
    --output-dir tracking_output \
    --conf 0.3
```

### Parameters

- `video_path`: Path to the input video file (required)
- `--model-path`: Path to YOLO model file (default: `yolo11m.pt`)
- `--output-dir`: Directory for output files (default: `tracking_output`)
- `--conf`: Confidence threshold for detection (default: 0.3)

## How It Works

### 1. Video Processing
- Processes video frames with adaptive skipping for performance
- Uses YOLOv11 to detect vehicles (cars, trucks, motorcycles, bicycles)
- Applies ByteTrack for robust multi-object tracking

### 2. Vehicle Filtering
A vehicle is considered "unique" and worth saving if it meets these criteria:
- **Movement Pattern**: Moving away from camera (getting smaller) and upward
- **Size Threshold**: Occupies at least 10% of screen area
- **Fallback Rule**: Appears in the bottom half of the frame

### 3. Output Generation
For each unique vehicle:
- **License Plate Found**: Saves the best frame as JPG in `counted_cars/`
- **No License Plate**: Saves a review clip (MP4) in `review/`
- **Analysis**: Optional OpenAI analysis for vehicle details

### 4. Deduplication
- 3-second cooldown prevents saving the same vehicle multiple times
- Grace period handles temporary occlusions

## Output Structure

```
tracking_output/
├── video_name/
│   ├── counted_cars/
│   │   ├── 00-01-09_car_3.jpg
│   │   ├── 00-02-01_car_7.jpg
│   │   └── ...
│   ├── review/
│   │   ├── 00-03-37_car_10_review.mp4
│   │   └── ...
│   └── vehicle_analysis_results.csv
```

## Configuration

### ByteTrack Configuration (`bytetrack.yaml`)
The tracker behavior can be customized by modifying `bytetrack.yaml`:
- Track buffer length
- High/low threshold ratios
- Match thresholds
- Frame rate settings

### Model Selection
- **YOLOv11n**: Fastest, lower accuracy
- **YOLOv11s**: Balanced speed/accuracy
- **YOLOv11m**: Higher accuracy, slower (default)

## Performance Optimization

- **Hardware Acceleration**: Automatically uses CUDA/MPS if available
- **Frame Skipping**: Adaptive skipping based on vehicle presence
- **Memory Management**: Efficient buffer management for long videos

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure `yolo11m.pt` and `license_plate_detector.pt` are in the project directory
2. **CUDA errors**: The system will fall back to CPU if GPU is unavailable
3. **Memory issues**: Reduce frame resolution or use a smaller YOLO model

### Performance Tips

- Use SSD storage for faster video I/O
- Ensure sufficient RAM (8GB+ recommended)
- Close other applications during processing

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv11
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for tracking algorithm
- [OpenAI](https://openai.com/) for vehicle analysis capabilities 