from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from .database import Base

class DetectionResult(Base):
    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    class_name = Column(String)
    confidence = Column(Float)
    x_min = Column(Float)
    y_min = Column(Float)
    x_max = Column(Float)
    y_max = Column(Float)

class MotionLog(Base):
    __tablename__ = "motion_log"

    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(Integer)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    max_count = Column(Integer)