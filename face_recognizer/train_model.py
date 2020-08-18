import cv2
import os
import numpy as np
import image
from PIL import  Image
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
def getImageAndLabels(path):
    #Lấy tất cả file trong thư mục
    imagePaths  = [os.path.join(path,f) for f in os.listdir(path)]
    #create empth face list
    faceSamples = []
    #create empty ID list
    Ids = []
    #Lặp thông qua tất cả ảnh trong đường dẫn và load ids và ảnh
    for imagePath in imagePaths:
        if(imagePath[-3:]=='jpg'):
            print(imagePath[-3:])
            #load va chuyen anh ve anh xam
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage,'uint8')
            #getting the id from the image
            Id = int(os.path.split(imagePath)[-1].split('.')[1])
            #extect the face from the training image sample
            faces = detector.detectMultiScale(imageNp)
            #if a face is there the append that in the list as well as Id of it
            for (x,y,w,h) in faces:
                faceSamples.append(imageNp[y:y+h,x:x+w])
                Ids.append(Id)
    return faceSamples,Ids
#Lấy các khuôn mặt và id từ thư mục dataSet
faceSamples, Ids = getImageAndLabels('dataSet')
#Train model để trích xuất đặc trưng của các khuôn mặt và gán với từng nhân viên
recognizer.train(faceSamples,np.array(Ids))
#Save model
recognizer.save('recognizer/trainner.yml')
print("Trained!")
