import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml') # Class đã train trước về face
watch_cascade = cv2.CascadeClassifier('cascades/data/cascade_watch.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Recognizers/face-trainner.yml")

labels = {"person_name": 1}
with open("Pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

frame = cv2.imread('C:\\Users\\tetan\\OneDrive\\Documents\\PyThon\\FaceId\\108.jpg')
#frame = cv2.resize(frame,(1280,720))

gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3) #hàm phát hiện ra vị trí khuôn mặt trên ảnh
watch = watch_cascade.detectMultiScale(gray, scaleFactor = 4, minNeighbors = 4)

for (x,y,w,h) in watch:
    roi_gray_watch = gray[y:y+h, x:x+w]
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)

    cv2.imwrite("watch_image.jpg",roi_gray_watch)

for (x, y, w, h) in faces:
    #print(x,y,w,h) # vị trí khuôn mặt trên ảnh
        
    roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end) # cắt ảnh chứa khuôn mặt
        
    roi_color = frame[y:y+h, x:x+w]
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    id_, conf = recognizer.predict(roi_gray) # dự đoán id_ và độ chính xác của ảnh từ camera và ảnh đã train

    if conf>=4 and conf <= 85:
        print(id_)
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = labels[id_]
        color = (255, 0, 0)
        stroke = 2
        cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

    cv2.imwrite("my-image.png",roi_gray) # lưu lại ảnh chứa khuôn mặt vừa cắt
    cv2.imshow('result',roi_gray)

cv2.imshow('frame',frame)
cv2.waitKey(0)
cv2.destroyAllWindows()