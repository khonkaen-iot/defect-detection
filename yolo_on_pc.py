import cv2
from ultralytics import YOLO 
import numpy as np

model = YOLO('best.pt')

cap = cv2.VideoCapture(1) # your vedio source 

while cap.isOpened():
    ret , frame = cap.read()
    
    results = model(frame)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype='int')
    classes = np.array(result.boxes.cls.cpu(),dtype='int')
    confidences = np.array(result.boxes.conf.cpu())
    class_names = result.names
    class_name_list = [class_names[i] for i in classes]

    conf_threshold = 0.5

    for cls, bbox , conf in zip(classes,bboxes, confidences):
        if conf > conf_threshold:
            x1,y1,x2,y2 = bbox
            label = class_names[cls]
            cv2.rectangle(frame, (x1,y1),(x2,y2),(255,0,0),3)
            cv2.putText(frame, label, (x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,2, (255,0,0),2)
    
    cv2.imshow('Screw Production Line no.1 ', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()