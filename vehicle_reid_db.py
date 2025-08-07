import sqlite3
import numpy as np
import json
from datetime import datetime

class VehicleReIDDatabase:
    def __init__(self, db_path='vehicle_reid.db'):
        self.db_path = db_path
        self._setup_db()

    def _setup_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicles (
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
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_global_id ON vehicles(global_id)
        ''')
        conn.commit()
        conn.close()

    def save_vehicle(self, camera_id, track_id, global_id, frame_number, bbox, features, confidence):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        features_json = json.dumps(features.tolist())
        timestamp = datetime.now().isoformat()
        cursor.execute('''
            INSERT INTO vehicles (camera_id, track_id, global_id, frame_number, timestamp, bbox_x1, bbox_y1, bbox_x2, bbox_y2, features, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (camera_id, str(track_id), str(global_id), frame_number, timestamp, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), features_json, confidence))
        conn.commit()
        conn.close()

    def get_all_features_by_camera(self, camera_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT track_id, features FROM vehicles WHERE camera_id=?', (camera_id,))
        data = cursor.fetchall()
        conn.close()
        features_dict = {}
        for track_id, features_json in data:
            features = np.array(json.loads(features_json))
            if track_id not in features_dict:
                features_dict[track_id] = []
            features_dict[track_id].append(features)
        return features_dict

    def get_all_features(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT camera_id, track_id, features FROM vehicles')
        data = cursor.fetchall()
        conn.close()
        features_dict = {}
        for camera_id, track_id, features_json in data:
            features = np.array(json.loads(features_json))
            if camera_id not in features_dict:
                features_dict[camera_id] = {}
            if track_id not in features_dict[camera_id]:
                features_dict[camera_id][track_id] = []
            features_dict[camera_id][track_id].append(features)
        return features_dict

    def get_features_for_matching(self, camera_id):
        # Returns a list of (track_id, features) for matching
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT track_id, features FROM vehicles WHERE camera_id=?', (camera_id,))
        data = cursor.fetchall()
        conn.close()
        return [(track_id, np.array(json.loads(features_json))) for track_id, features_json in data]
