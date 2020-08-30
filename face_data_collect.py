#Read and show video stream, capture images
#Detect Faces and show bounding boxes
#Flatten the largest face image and save in a numpy array
#Repeat the above for multiple people to generate training data



import cv2
import numpy as np

#init camera
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_data = []

skip=0

data_set_path = './data/'

file_name = input("Enter your name : ")

while True:

    ret,frame = cap.read()

    if ret == False:
        continue

    face_section=frame

    faces = face_cascade.detectMultiScale(frame,1.3,5)
    faces = sorted(faces,key=lambda f:f[2]*f[3])
 
#Pick from the last face beacause it is the largest face according to area
    for face in faces[-1:]:
        (x,y,w,h)=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        
        #Extract (Crop out the required Face) : Region Of Interest
        offset=10
        face_section=frame[y-offset:y+offset+h,x-offset:x+offset+w]
        face_section=cv2.resize(face_section,(100,100))

        skip+=1
        #store every 10th face
        if skip%10 == 0:
            face_data.append(face_section)
            print(len(face_data))
            
    
    
    cv2.imshow("Frame",frame)
    cv2.imshow("Face Section",face_section)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

#Convert our face list in numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#Save this data into the file system
np.save(data_set_path+file_name+'.npy',face_data)
print("Data Successfully saved at "+data_set_path+file_name+'.npy')


cap.release()
cv2.destroyAllWindows()