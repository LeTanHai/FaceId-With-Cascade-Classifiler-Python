import cv2
import numpy as np
import os
from PIL import Image
import pickle

current_id = 0
label_ids = {}
x_train = []
y_labels = []

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # trả về đường dẫn đến thư mục chứa file hiện hành
image_dir = os.path.join(BASE_DIR, "Images")  # Nối đường dẫn BASE_DIR với Images

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml') # Class đã train trước về face
recognizer = cv2.face.LBPHFaceRecognizer_create() # hàm này sẽ tạo file .yml để lưu dữ liệu training sử dụng để nhận diện khuôn mặt


for root,dirs,files in os.walk(image_dir):  # os.walk(image_dir) duyệt cây thư mục trong đường dẫn
											# root trả về đường dẫn đến các thư mục
											# files trả về các file trong thư mục tính từ thư mục cao nhất trong đường dẫn
    for file in files:
    	if file.endswith("png") or file.endswith("jpg"):
    		path = os.path.join(root,file)
    		label = os.path.basename(os.path.dirname(path)).lower() # os.path.basename() trả về tên file nằm ở cuối đường dẫn
    														# os.path.dirname() trả về đường dẫn chỉ chứa tên thư mục
    		#print(label)
    		if not label in label_ids:            # tạo 1 dictionary chứa tên thư mục và id 
    			label_ids[label] = current_id
    			current_id += 1
    		id_ = label_ids[label]				  # trong 1 thư mục dù có nhiều ảnh thì id vẫn cố định vè id này dùng để xác định tên thư mục
    		print(label_ids)
            
    		pil_image = Image.open(path).convert("L")   # Mở ảnh từ đường dẫn và chuyển về grayscale
    		image_array = np.array(pil_image,"uint8")   # chuyển ảnh về dạng mảng Numpy kiểu uint8
    		#print(image_array)

    		faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.1, minNeighbors=3) #hàm phát hiện ra vị trí khuôn mặt trên ảnh
    		for (x,y,w,h) in faces:
    			roi = image_array[y:y+h, x:x+w]
    			x_train.append(roi)
    			y_labels.append(id_)

    		#print(label_ids)
    		#print(id_)
#print(x_train)
#print(y_labels)

with open("Pickles/face-labels.pickle", 'wb') as f:   # lưu dictionay y_labels vào file face-labels.pickle
	pickle.dump(label_ids, f)                         # hàm dump() dùng ghi dữ liệu vào file, module pickles ghi dữ liệu dưới dạng kí hiệu
													  # khi đọc nó sẽ chuyển lại dạng text cho chúng ta

recognizer.train(x_train, np.array(y_labels))    # tạo ra 1 set gồm data và labels
recognizer.save("Recognizers/face-trainner.yml") # lưu vào file yml