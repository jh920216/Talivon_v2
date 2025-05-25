import os
import cv2
import threading

from dotenv import load_dotenv
from datetime import datetime, timedelta

from rtsp.rtsp_stream import get_rtsp_stream
from rtsp.frame_processor import detect_person, detect_paying
from db.database import get_db
from db.crud import get_all_detections, delete_detection, insert_motion_period

# load_dotenv()
load_dotenv()