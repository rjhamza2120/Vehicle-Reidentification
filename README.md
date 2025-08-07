# Vehicle Re-Identification System

A comprehensive vehicle re-identification system that tracks and re-identifies vehicles across multiple camera views using deep learning techniques. This system combines YOLO-based vehicle detection, DeepSORT tracking, and MobileNetV2 feature extraction to achieve accurate vehicle re-identification.

## 🚗 Features

- **Multi-Camera Vehicle Tracking**: Track vehicles across different camera views with DeepSORT
- **Deep Learning-Based Detection**: YOLOv8-based vehicle detection with high accuracy
- **Advanced Feature Extraction**: MobileNetV2 with multi-view augmentation and CLAHE enhancement
- **Robust Re-identification**: Cosine similarity-based matching with temporal consistency
- **Real-time Visualization**: Live tracking display with bounding boxes and match information
- **Database Storage**: SQLite database for storing vehicle features and metadata
- **Performance Monitoring**: Real-time FPS and tracking statistics
- **GPU Acceleration**: CUDA support for faster processing

## 🏗️ System Architecture

The system consists of two main components:

### 1. VehicleReIDCamera Class
- **Vehicle Detection**: YOLOv8 model for detecting vehicles (car, motorcycle, bus, truck)
- **Feature Extraction**: MobileNetV2 with multi-angle views and CLAHE enhancement
- **Tracking**: DeepSORT with optimized parameters for vehicle tracking
- **Re-identification**: Cosine similarity matching with temporal consistency checks

### 2. VehicleReIDDatabase Class
- **SQLite Database**: Stores vehicle features, metadata, and re-identification matches
- **Feature Storage**: JSON-serialized feature vectors with timestamps
- **Query Interface**: Methods for retrieving features by camera and track ID

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for acceleration)
- Webcam or video files for testing

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Vehicle-Reidentification
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Basic Usage

Run the system with two video files (Camera A and Camera B):

```bash
python vehicle_reid_system_fixed.py
```

### Configuration

The system can be configured with the following parameters:

```python
# Initialize the system
system = VehicleReIDSystem(
    db_path='vehicle_reid.db',
    model_path='yolov8n.pt'
)

# Run with video files
system.run('path/to/camera_a.mp4', 'path/to/camera_b.mp4')
```

### Camera Configuration

Each camera can be configured with:

```python
camera = VehicleReIDCamera(
    camera_id='A',                    # Camera identifier
    db_path='vehicle_reid.db',        # Database path
    model_path='yolov8n.pt',          # YOLO model path
    confidence_threshold=0.5,          # Detection confidence
    feature_extraction_interval=2,     # Feature extraction frequency
    similarity_threshold=0.4,          # Re-identification threshold
    min_track_length=3                 # Minimum track length for re-ID
)
```

## 🔧 Key Components

### 1. Vehicle Detection
- **Model**: YOLOv8 with configurable confidence threshold
- **Classes**: Supports car, motorcycle, bus, truck detection
- **Filtering**: Minimum size threshold (40x40 pixels) to avoid small detections
- **Resolution**: 640x640 input resolution for optimal detection

### 2. Feature Extraction
- **Backbone**: MobileNetV2 with ImageNet pretrained weights
- **Multi-view Augmentation**: 5 different rotation angles (-15°, -7.5°, 0°, 7.5°, 15°)
- **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization
- **Horizontal Flipping**: Additional augmentation for robustness
- **Quality Assessment**: Feature normalization and quality checks

### 3. Vehicle Tracking
- **Algorithm**: DeepSORT with optimized parameters
- **Parameters**:
  - `max_age=50`: Longer tracking persistence
  - `n_init=3`: More frames needed to confirm track
  - `max_cosine_distance=0.3`: Feature matching threshold
  - `nn_budget=100`: Keep more samples per track
- **Movement Detection**: Tracks vehicle movement with configurable threshold
- **Track Management**: Automatic cleanup of old tracks

### 4. Re-identification
- **Similarity Metric**: Cosine similarity with L2 normalization
- **Temporal Consistency**: Frame difference constraints (10 seconds at 30fps)
- **Quality Checks**: Minimum consistent matches (3) for reliable matching
- **Clear Winner Logic**: Prevents ambiguous matches
- **Movement Constraint**: Only matches moving vehicles in Camera B

### 5. Database Schema

**vehicles table**:
```sql
CREATE TABLE vehicles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    track_id TEXT NOT NULL,
    global_id TEXT NOT NULL,
    frame_number INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    bbox_x1 INTEGER NOT NULL,
    bbox_y1 INTEGER NOT NULL,
    bbox_x2 INTEGER NOT NULL,
    bbox_y2 INTEGER NOT NULL,
    features TEXT NOT NULL,
    confidence REAL NOT NULL
)
```

## 📊 Performance Features

### Real-time Processing
- **Detection**: ~30 FPS on GPU, ~15 FPS on CPU
- **Feature Extraction**: ~25 FPS on GPU, ~10 FPS on CPU
- **Tracking**: Minimal overhead with DeepSORT
- **Re-identification**: Real-time matching with database queries

### Visualization
- **Bounding Boxes**: Color-coded by track ID or matched ID
- **Labels**: Display track ID, matched ID, and similarity score
- **Frame Information**: Camera ID, frame count, active tracks
- **Re-identification Status**: Number of re-identified vehicles

## 🎯 Advanced Features

### 1. Multi-view Feature Extraction
```python
# Multiple angle views for robust features
angles = [-15, -7.5, 0, 7.5, 15]
for angle in angles:
    # Rotate and enhance image
    # Extract features for each view
    # Combine features for final representation
```

### 2. Movement-based Re-identification
```python
# Only match moving vehicles in Camera B
if not is_moving:
    return None, 0.0
```

### 3. Temporal Consistency
```python
# Frame difference constraint
max_frame_difference = 300  # 10 seconds at 30fps
if frame_diff > max_frame_difference:
    continue
```

### 4. Quality Assessment
```python
# Feature quality checks
feature_norm = np.linalg.norm(features)
if 0.9 <= feature_norm <= 1.1:
    # Store high-quality features
```

## 🔍 Example Output

The system provides:
- **Processed Videos**: With bounding boxes, labels, and match information
- **Database**: Vehicle features and metadata storage
- **Real-time Statistics**: Frame count, active tracks, re-identification count
- **Match History**: Temporal tracking of re-identification matches

## 📝 Dependencies

The system requires the following packages (see `requirements.txt`):

```
opencv-python
ultralytics
torch
torchreid
torchvision
numpy
accelerate
pillow
gdown
tensorboard
deep_sort_realtime
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [DeepSORT](https://github.com/mikel-brostrom/yolov8_tracking) for tracking
- [PyTorch](https://pytorch.org/) for deep learning framework
- [OpenCV](https://opencv.org/) for computer vision operations

## 📞 Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the code comments for implementation details
- Review the database schema for data structure

## 🔄 Version History

- **v1.0.0**: Initial release with integrated architecture
  - YOLOv8-based vehicle detection
  - MobileNetV2 feature extraction with multi-view augmentation
  - DeepSORT tracking with optimized parameters
  - SQLite database storage
  - Real-time visualization and re-identification

---

**Note**: This system is designed for research and educational purposes. For production use, additional testing and validation are recommended.

