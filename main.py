import cv2
import numpy as np
from ultralytics import YOLO
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from vehicle_reid_db import VehicleReIDDatabase
from collections import defaultdict

class VehicleReIDCamera:
    def __init__(self, camera_id, db_path, model_path='yolov8n.pt', 
                 confidence_threshold=0.5, feature_extraction_interval=2, 
                 similarity_threshold=0.4, min_track_length=3):
        self.camera_id = camera_id  
        self.db = VehicleReIDDatabase(db_path)
        self.similarity_threshold = similarity_threshold
        self.min_track_length = min_track_length
        self.matched_camera_a_ids = set()  
        self.movement_threshold = 30  
        self.max_angle_diff = 30 
        
        # YOLO model for vehicle detection
        self.model = YOLO(model_path)
        self.model.conf = confidence_threshold
        
        # Initialize DeepSORT 
        self.tracker = DeepSort(
            max_age=50,        
            n_init=3,          
            nms_max_overlap=0.7,
            max_cosine_distance=0.3,
            nn_budget=100     
        )
        
        # Vehicle classes in COCO dataset
        self.vehicle_class_ids = [2,3,5,7]  # car, motorcycle, bus, truck
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.feature_model = models.mobilenet_v2(pretrained=True)

        self.feature_model.eval()
        self.feature_model = self.feature_model.to(self.device)
        
        # Image preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),  
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.feature_extraction_interval = feature_extraction_interval
        self.track_features = defaultdict(list) 
        self.track_metadata = defaultdict(dict)  
        self.reid_matches = {}  
        self.colors = self._generate_colors(100)
        
    def _generate_colors(self, num_colors):
        np.random.seed(42)
        return [tuple(map(int, color)) for color in np.random.randint(0, 255, (num_colors, 3))]

    def extract_features(self, vehicle_image):
        try:
            if vehicle_image.shape[0] < 128 or vehicle_image.shape[1] < 128:
                return None
            
            vehicle_image_rgb = cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2RGB)
            
            angle_views = []
            angles = [-15, -7.5, 0, 7.5, 15]  
            
            for angle in angles:
                height, width = vehicle_image_rgb.shape[:2]
                rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
                rotated_img = cv2.warpAffine(vehicle_image_rgb, rotation_matrix, (width, height))
                
                lab = cv2.cvtColor(rotated_img, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                enhanced_img = cv2.merge((cl,a,b))
                enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2RGB)
                
                pil_image = Image.fromarray(enhanced_img)
                angle_views.append(pil_image)
                
                # Add horizontally flipped version for each angle
                flipped_img = Image.fromarray(cv2.flip(enhanced_img, 1))
                angle_views.append(flipped_img)
            
            # Additional preprocessing for better feature extraction
            enhanced_transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # Process all angle views
            all_features = []
            
            for view in angle_views:
                input_tensor = enhanced_transforms(view).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    # Extract features for this view
                    features = self.feature_model(input_tensor)
                    if isinstance(features, tuple):
                        features = features[0]
                    
                    # Process and normalize features
                    features = features.squeeze().cpu().numpy()
                    if len(features.shape) > 1:
                        features = features.reshape(-1)
                    
                    # Apply L2 normalization
                    features = features / (np.linalg.norm(features) + 1e-6)
                    
                    # Quality check for this view
                    if not np.isnan(features).any() and not np.isinf(features).any():
                        all_features.append(features)
            
            if not all_features:
                return None
                
            # Combine features from all views
            combined_features = np.mean(all_features, axis=0)
            combined_features = combined_features / np.linalg.norm(combined_features)
            
            return combined_features
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def detect_vehicles(self, frame):
        results = self.model(frame, verbose=False, imgsz=640)  
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if class_id in self.vehicle_class_ids and confidence >= self.model.conf:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        width = x2 - x1
                        height = y2 - y1
                        if width < 40 or height < 40:  
                            continue
                            
                        detections.append([x1, y1, x2, y2, confidence])
        
        return detections

    def cleanup_old_tracks(self, current_frame):
        """Remove tracks that haven't been seen for a while"""
        max_age = 150  
        tracks_to_remove = []
        
        for track_id, metadata in self.track_metadata.items():
            if current_frame - metadata['last_seen'] > max_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.track_metadata[track_id]
            del self.track_features[track_id]
            if track_id in self.reid_matches:
                del self.reid_matches[track_id]

    def update_track_info(self, track_id, features, frame_number, position=None):
        """Update track information and features with quality checks and movement tracking"""
        if track_id not in self.track_metadata:
            self.track_metadata[track_id] = {
                'first_seen': frame_number,
                'last_seen': frame_number,
                'frames_tracked': 1,
                'avg_feature_quality': 0.0,
                'positions': [position] if position else [],
                'is_moving': False
            }
        else:
            self.track_metadata[track_id]['last_seen'] = frame_number
            self.track_metadata[track_id]['frames_tracked'] += 1
            
            # Track movement
            if position:
                positions = self.track_metadata[track_id]['positions']
                positions.append(position)
                
                # Keep only recent positions
                if len(positions) > 30:  
                    positions.pop(0)
                
                # Check if vehicle is moving
                if len(positions) >= 2:
                    total_movement = sum(abs(positions[-1][0] - pos[0]) + abs(positions[-1][1] - pos[1]) 
                                      for pos in positions[:-1]) / len(positions[:-1])
                    self.track_metadata[track_id]['is_moving'] = total_movement > self.movement_threshold

        # Quality check on features
        feature_norm = np.linalg.norm(features)
        if 0.9 <= feature_norm <= 1.1:  # Check if features are properly normalized
            # Store feature with timestamp
            self.track_features[track_id].append((features, frame_number))
            
            # Keep most recent features and maintain temporal distribution
            if len(self.track_features[track_id]) > 15:  # Increased history
                # Sort by timestamp and keep evenly distributed samples
                sorted_features = sorted(self.track_features[track_id], key=lambda x: x[1])
                indices = np.linspace(0, len(sorted_features)-1, 15, dtype=int)
                self.track_features[track_id] = [sorted_features[i] for i in indices]

        # Cleanup old tracks periodically
        if frame_number % 30 == 0: 
            self.cleanup_old_tracks(frame_number)

    def _is_track_already_matched(self, track_id):
        """Check if a track from Camera A is already matched to a Camera B track"""
        for b_track, (a_track, _) in self.reid_matches.items():
            if a_track == track_id:
                return True
        return False

    def find_best_match(self, current_features, cam_a_features_dict, current_frame, is_moving=False):
        """Find best matching track from Camera A with improved matching logic"""
        matches = []
        max_frame_difference = 300  
        min_similarity = self.similarity_threshold
        min_consistent_matches = 3  
        
        # Only consider moving vehicles in Camera B for re-identification
        if not is_moving:
            return None, 0.0
        
        # Ensure current_features is 1D and valid
        if current_features is None or len(current_features.shape) > 1:
            if current_features is not None:
                current_features = current_features.reshape(-1)
            else:
                return None, 0.0
        
        # Compare with all Camera A tracks
        for track_id, features_data in cam_a_features_dict.items():
            # Skip if track is already matched with higher confidence
            if self._is_track_already_matched(track_id):
                continue
            
            # Check temporal constraint with dynamic threshold
            if track_id in self.track_metadata:
                frame_diff = abs(self.track_metadata[track_id]['last_seen'] - current_frame)
                if frame_diff > max_frame_difference:
                    continue
            
            # Calculate similarity with all features of this track
            similarities = []
            timestamps = []
            
            for feat_tuple in features_data:
                feat_a = feat_tuple[0]  # Feature vector
                timestamp = feat_tuple[1]  # Timestamp
                
                if feat_a is not None and feat_a.size > 0:
                    try:
                        # Ensure feat_a is 1D
                        if len(feat_a.shape) > 1:
                            feat_a = feat_a.reshape(-1)
                        
                        # Enhanced similarity calculation
                        feat_a_norm = feat_a / (np.linalg.norm(feat_a) + 1e-6)
                        current_features_norm = current_features / (np.linalg.norm(current_features) + 1e-6)
                        
                        # Compute similarity with additional checks
                        sim = np.dot(current_features_norm, feat_a_norm)
                        
                        # Additional validation
                        if not np.isnan(sim) and not np.isinf(sim) and -1.0 <= sim <= 1.0:
                            similarities.append(float(sim))
                            timestamps.append(timestamp)
                            
                    except Exception as e:
                        print(f"Similarity calculation error: {e}")
                        continue
            
            # Enhanced matching logic with temporal consistency
            if len(similarities) >= min_consistent_matches:
                sim_time_pairs = sorted(zip(similarities, timestamps), reverse=True)
                top_similarities = [s for s, _ in sim_time_pairs[:5]]
                
                # Calculate various statistics for robust matching
                avg_sim = np.mean(top_similarities)
                max_sim = np.max(top_similarities)
                min_sim = np.min(top_similarities)
                std_sim = np.std(top_similarities)
                
                is_consistent = std_sim < 0.1  
                has_enough_high_scores = sum(s > self.similarity_threshold for s in top_similarities) >= min_consistent_matches
                
                # Calculate confidence score based on multiple factors
                confidence_score = avg_sim * (1.0 - std_sim) * (1.0 if is_consistent else 0.5)
                
                if (confidence_score >= min_similarity and 
                    has_enough_high_scores and 
                    is_consistent and 
                    max_sim - min_sim < 0.2): 
                    
                    matches.append((track_id, confidence_score))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        
        if matches:
            best_match = matches[0]
            if (best_match[1] > self.similarity_threshold + 0.1 and  
                (len(matches) == 1 or best_match[1] - matches[1][1] > 0.1)):  
                return best_match
        
        return None, 0.0

    def process_video(self, video_path, cam_a_features_dict=None, assign_global_ids=False):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
            
        frame_count = 0
        window_name = f"Camera {self.camera_id} - Vehicle Re-ID"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print(f"Processing video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            detections = self.detect_vehicles(frame)
            tracks = self.track_vehicles(frame, detections)
            
            # Process each track
            annotated_frame = frame.copy()
            active_tracks = []
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                track_id = track.track_id
                active_tracks.append(track_id)
                
                # Get bounding box
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                
                # Extract vehicle crop
                vehicle_crop = frame[y1:y2, x1:x2]
                
                # Process features and re-identification
                if frame_count % self.feature_extraction_interval == 0:
                    features = self.extract_features(vehicle_crop)
                    if features is not None:
                        self.update_track_info(track_id, features, frame_count)
                        
                        # Re-identification for Camera B
                        if assign_global_ids and cam_a_features_dict and self.camera_id == 'B':
                            if self.track_metadata[track_id]['frames_tracked'] >= self.min_track_length:
                                try:
                                    center_x = (x1 + x2) / 2
                                    center_y = (y1 + y2) / 2
                                    current_pos = (center_x, center_y)
                                    
                                    self.update_track_info(track_id, features, frame_count, current_pos)
                                    
                                    update_match = False
                                except Exception as e:
                                    print(f"Error in re-identification process: {e}")
                                    continue
                                
                                best_id, similarity = self.find_best_match(
                                    features, cam_a_features_dict, frame_count,
                                    is_moving=self.track_metadata[track_id]['is_moving']
                                )
                                
                                # Update match if:
                                # 1. No previous match exists
                                # 2. Camera A ID hasn't been matched yet, or this is a better match
                                # 3. Similarity is above threshold
                                
                                if (best_id is not None and 
                                    similarity > self.similarity_threshold and 
                                    (best_id not in self.matched_camera_a_ids or
                                     similarity > self.similarity_threshold + 0.2)):
                                        
                                        if track_id not in self.reid_matches:
                                            update_match = True
                                        else:
                                            old_id, old_sim = self.reid_matches[track_id]
                                            if similarity > old_sim + 0.15:  
                                                update_match = True
                                                
                                if update_match and best_id is not None:
                                    self.reid_matches[track_id] = (best_id, similarity)
                                    self.matched_camera_a_ids.add(best_id)  
                                    print(f"Re-ID: B{track_id} -> A{best_id} (sim={similarity:.3f})")
                                    
                                    if 'match_history' not in self.track_metadata[track_id]:
                                        self.track_metadata[track_id]['match_history'] = []
                                    self.track_metadata[track_id]['match_history'].append((frame_count, best_id, similarity))
                
                if assign_global_ids and self.camera_id == 'B' and track_id in self.reid_matches:
                    matched_id, sim = self.reid_matches[track_id]
                    label = f"A{matched_id} ({sim:.2f})"
                    color = self.colors[hash(str(matched_id)) % len(self.colors)]
                else:
                    label = f"{self.camera_id}{track_id}"
                    color = self.colors[hash(str(track_id)) % len(self.colors)]
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_frame, 
                            (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), 
                            color, -1)
                cv2.putText(annotated_frame, label, 
                          (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, (255, 255, 255), 2)
            
            info_text = [
                f"Camera: {self.camera_id}",
                f"Frame: {frame_count}",
                f"Active Tracks: {len(active_tracks)}",
            ]
            if self.camera_id == 'B':
                info_text.append(f"Re-identified: {len(self.reid_matches)}")
            
            y = 30
            for text in info_text:
                cv2.putText(annotated_frame, text, (10, y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y += 25
            
            cv2.imshow(window_name, annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyWindow(window_name)
        print(f"Processed {frame_count} frames for Camera {self.camera_id}")

    def track_vehicles(self, frame, detections):
        if not detections:
            return self.tracker.update_tracks([], frame=frame)
            
        detection_list = []
        for x1, y1, x2, y2, conf in detections:
            width = x2 - x1
            height = y2 - y1
            detection_list.append(([x1, y1, width, height], conf, 'vehicle'))
            
        return self.tracker.update_tracks(detection_list, frame=frame)

class VehicleReIDSystem:
    def __init__(self, db_path='vehicle_reid.db', model_path='yolov8n.pt'):
        self.db_path = db_path
        self.model_path = model_path
        
        print("Initializing Vehicle Re-ID System...")
        self.cam_a = VehicleReIDCamera('A', db_path, model_path)
        self.cam_b = VehicleReIDCamera('B', db_path, model_path)
        print("System initialized.")

    def run(self, video_path_a, video_path_b):
        # Process Camera A first to build feature database
        print("\nProcessing Camera A video...")
        self.cam_a.process_video(video_path_a)
        
        # Get features from Camera A and convert to compatible format
        features_dict = {}
        for track_id, feature_data in self.cam_a.track_features.items():
            # Only include tracks that have valid features
            valid_features = [(feat[0], feat[1]) for feat in feature_data if feat[0] is not None and feat[0].size > 0]
            if valid_features:
                features_dict[track_id] = valid_features
        
        # Process Camera B with re-identification
        print("\nProcessing Camera B video with re-identification...")
        self.cam_b.process_video(video_path_b, features_dict, assign_global_ids=True)
        
        print("\nRe-identification complete!")
        print(f"Total tracks re-identified: {len(self.cam_b.reid_matches)}")

if __name__ == "__main__":
    system = VehicleReIDSystem()
    system.run('Data/3.mp4', 'rtsp://admin:ad123456@192.168.55.252:554/0')