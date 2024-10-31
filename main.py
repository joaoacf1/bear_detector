import cv2
import cvzone
from ultralytics import YOLO
import winsound
import threading

video = cv2.VideoCapture('assets/video01.mp4')
model = YOLO('yolov8l.pt')
controlAlarm = False

def alarm():
    global controlAlarm
    for _ in range(5):
        winsound.Beep(2500, 500)
    controlAlarm = False

while video.isOpened():
    check, img = video.read()
    if not check:
        break  
    
    img = cv2.resize(img, (640, 360))
    results = model.predict(img, conf=0.5)[0] 
    
    for data in results.boxes:
        x, y, w, h = map(int, data.xyxy[0]) 
        cls = int(data.cls[0])
        
        if cls == 21:
            cv2.rectangle(img, (x, y), (w, h), (255, 0, 255), 5)
            cvzone.putTextRect(img, 'BEAR!!!', pos=(105, 65), colorR=(0, 0, 255))
         
            if not controlAlarm:
                controlAlarm = True
                threading.Thread(target=alarm).start()
                
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

video.release()
cv2.destroyAllWindows()