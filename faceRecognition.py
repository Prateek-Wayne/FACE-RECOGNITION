import face_recognition
import cv2
face1=cv2.imread('Attendance/chintu1.png')
face2=cv2.imread(('FR/chintu2.jpg'))
# Resizing
size=(550,550)
# face1=cv2.resize(face1,size)
face1=cv2.resize(face1,(550,550),cv2.INTER_AREA)
face2=cv2.resize(face2,(550,550),cv2.INTER_AREA)

# Converting BGR to RGB
face1=cv2.cvtColor(face1,cv2.COLOR_BGR2RGB)
face2=cv2.cvtColor(face2,cv2.COLOR_BGR2RGB)

#Detecting Face for test Image 1
face_loc=face_recognition.face_locations(face1)[0]
# Encoding face with 128 points
face_encode=face_recognition.face_encodings(face1)[0]

# Creating Rectangle Around Face
cv2.rectangle(face1,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(0,255,0),2)

#Detecting Face for test Image 1
face_loc_test=face_recognition.face_locations(face2)[0]
# Encoding face with 128 points
face_encode_test=face_recognition.face_encodings(face2)[0]

# Creating Rectangle Around Face
cv2.rectangle(face2,(face_loc_test[3],face_loc_test[0]),(face_loc_test[1],face_loc_test[2]),(0,255,0),2)

# print(face_encode)
# print(face_loc)
# FInding Distance Between two Image Match OR Not
dis=face_recognition.face_distance([face_encode],face_encode_test)
result=face_recognition.compare_faces([face_encode],face_encode_test)

print(result)
cv2.putText(face2,f"{result}..matching{1-dis}",(100,100),cv2.FONT_HERSHEY_COMPLEX,1/2,(0,255,0),1)
cv2.imshow("Testing",face2)
cv2.imshow("Chintu1",face1)
cv2.waitKey(0)
