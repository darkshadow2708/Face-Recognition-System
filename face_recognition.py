import numpy as np

import cv2

import os

def distance(x1,x2):
    return np.sqrt(((x1-x2)**2).sum())

def knn(train,test,k=5):
    dist = []


    for i in range(train.shape[0]):
        #Get the vector ans label
        ix = train[i, :-1]
        iy = train[i,-1]
        #Compute distance from the test point
        d = distance(ix,test)
        dist.append([d,iy])
        #Sort based on distance and get the first k
    dk=sorted(dist,key=lambda x:x[0])[:k]
    #Retrieve the label
    labels = np.array(dk)[:,-1]
    #Get Frequency of Each Level
    output = np.unique(labels,return_counts=True)
    #Get the index of max frequency
    index = np.argmax(output[1])
    return output[0][index]


#init camera
cap = cv2.VideoCapture(0)

#Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

data_set_path = './data/'

face_data = []
labels = []

class_id=0 #label for the given file

names = {} #Mapping between id and name

#Data Preparation/Training data

for fx in os.listdir(data_set_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        print("Loaded" + fx)
        data_item = np.load(data_set_path + fx)
        face_data.append(data_item)


        #Create Labels for the Class
        target = class_id*np.ones((data_item.shape[0],))
        class_id+=1
        labels.append(target)


face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))
print(face_dataset.shape)
print(face_labels.shape)
train_set = np.concatenate((face_dataset,face_labels),axis=1)
print(train_set.shape)

#Testing Data
while True:
    ret,frame = cap.read()

    if ret==False:
        continue

    faces = face_cascade.detectMultiScale(frame,1.3,5)

    for face in faces:
        (x,y,w,h) = face

        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))

        out = knn(train_set,face_section.flatten())

        pred_name = names[int(out)]

        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow("Faces",frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()







