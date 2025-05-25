from sqlalchemy.orm import Session
from .models import DetectionResult
from db.models import MotionLog  # 감지 로그용 테이블 모델

def save_detection(db: Session, camera_id: str, detections: list):
    for det in detections:
        db_record = DetectionResult(
            camera_id=camera_id,
            class_name=det["name"],
            confidence=det["confidence"],
            x_min=det["x_min"],
            y_min=det["y_min"],
            x_max=det["x_max"],
            y_max=det["y_max"],
        )
        db.add(db_record)
    db.commit()

def get_all_detections(db: Session):
    return db.query(DetectionResult).all()

def delete_detection(db: Session, detection_id: int):
    record = db.query(DetectionResult).filter(DetectionResult.id == detection_id).first()
    if record:
        db.delete(record)
        db.commit()
        return True
    return False

def insert_motion_period(db: Session, store_id: str, start_time, end_time, max_count):
    log = MotionLog(store_id=store_id, start_time=start_time, end_time=end_time, max_count=max_count)
    db.add(log)
    db.commit()