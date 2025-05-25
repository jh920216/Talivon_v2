import cv2

def get_rtsp_stream(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("RTSP 스트림을 열 수 없습니다.")
        return None
    return cap
