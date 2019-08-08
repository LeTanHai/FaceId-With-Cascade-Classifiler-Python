import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml') # Class đã train trước về face
watch_cascade = cv2.CascadeClassifier('cascades/data/watch_cascade.xml')
casio_cascade = cv2.CascadeClassifier('cascades/data/Casio_cascade.xml')
iphone_cascade = cv2.CascadeClassifier('cascades/data/iphone_cascade.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Recognizers/face-trainner.yml")

labels = {"person_name": 1}
with open("Pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)
#frame = cv2.imread('C:\\Users\\tetan\\OneDrive\\Documents\\PyThon\\FaceId\\Images\\TanHai\\1.jpg')

while True:
    ret,frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5) #hàm phát hiện ra vị trí khuôn mặt trên ảnh
    #watch = watch_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors= 20)
    watch = watch_cascade.detectMultiScale(gray, scaleFactor=5, minNeighbors= 8)

    casio = casio_cascade.detectMultiScale(gray, scaleFactor=10, minNeighbors= 10)

    iphone = iphone_cascade.detectMultiScale(gray, scaleFactor=8, minNeighbors= 8)

    for (x, y, w, h) in casio:
    	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)


    for (x, y, w, h) in watch:
    	cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
    	roi_gray_watch = frame[y:y+h, x:x+w]

    for (x, y, w, h) in iphone:
    	cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)

    for (x, y, w, h) in faces:
        print(x,y,w,h) # vị trí khuôn mặt trên ảnh
        
        roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end) # cắt ảnh chứa khuôn mặt
        
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        id_, conf = recognizer.predict(roi_gray) # dự đoán id_ và độ chính xác của ảnh từ camera và ảnh đã train

        if conf>=45 and conf <= 85:
        	print(id_)
        	font = cv2.FONT_HERSHEY_SIMPLEX
        	name = labels[id_]
        	color = (255, 255, 255)
        	stroke = 2
        	cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        cv2.imwrite("my-image.png",roi_gray) # lưu lại ảnh chứa khuôn mặt vừa cắt
        cv2.imwrite("my-image_watch.png",roi_gray_watch)
    
    cv2.imshow('frame',frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()