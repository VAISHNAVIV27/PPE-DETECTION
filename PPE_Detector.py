from Detector import Yolo
import cv2
import time
import numpy as np

detector = Yolo(r"C:\Users\vaish_pz1\Desktop\PPE\INFERENCE\Ml_model\yolov3-tiny-obj.cfg",r"C:\Users\vaish_pz1\Desktop\PPE\INFERENCE\Ml_model\yolov3-tiny-obj_best.weights",["person","apron","without_apron","helmet","without_helmet"])
cap = cv2.VideoCapture(r"C:\Users\vaish_pz1\Desktop\PPE\Videos\PPE_2.mp4")
height = 540
width = 960
fps = 7
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
videosave = cv2.VideoWriter(r'C:\Users\vaish_pz1\Desktop\PPE\Inference.MP4',fourcc, fps, (width, height))

ret, frame = cap.read()
ft = 10



def draw_on_frame(frame, results):
    for cls, objs in results.items():
        for x1, y1, x2, y2 in objs:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255))
            cv2.putText(frame, cls, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), thickness=2)
            b = len(objs)

            if cls =="without_helmet" and "person":
                cv2.putText(frame,f"HELMET violation by {b} person",(1400,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0, 255), thickness=2)
            elif cls == "without_apron" and "person":
                cv2.putText(frame, f"Apron violation by {b} person", (1400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0, 255), thickness=2)
            elif cls == "without_apron" and "without_helmet":
                cv2.putText(frame, f"PPE violation by {b} person", (1400, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),
                            thickness=2)

    return frame

while (cap.isOpened()):
    start_time = time.time()
    ret, frame = cap.read()
    ret, frame = cap.read()
    results = detector.detect(frame, conf=0.7)
    print(results.items())
    frame = draw_on_frame(frame, results)
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    print(frame.shape)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(ft) & 0xFF
    if key == ord('q'):
        break
    videosave.write(frame)
cap.release()
videosave.release()
cv2.destroyAllWindows()



