import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
path="Attendance"
myList=os.listdir(path)
print(f'Path is :{myList}')
Images=[]
ImagesName=[]
for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    Images.append(curImg)
    ImagesName.append(os.path.splitext(cl)[0])

print(ImagesName)
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
def encodings(Images):
    encodingList=[]
    for i in Images:
        i=cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
        encodes=face_recognition.face_encodings(i)[0]
        encodingList.append(encodes)
    return encodingList
encodings_of_imagesKnown=encodings(Images)


print(f'lenght of encodingList: {len(encodings_of_imagesKnown)}')

# Capturing Frames from WebCam
capture=cv2.VideoCapture(0)
print("Entering While Loop-->")
while True:
    isTrue,frame=capture.read()
    frames=cv2.resize(frame,(0,0),None,0.60,0.60)
    faces_Curr_frame=face_recognition.face_locations(frames)
    face_Encodings_Known=face_recognition.face_encodings(frames,faces_Curr_frame)
    for encodes,faceLoc in zip(face_Encodings_Known,faces_Curr_frame):
        matches=face_recognition.compare_faces(encodings_of_imagesKnown,encodes)
        distance=face_recognition.face_distance(encodings_of_imagesKnown,encodes)

        bestMatchIndx =int(np.argmin(distance))
        if matches[bestMatchIndx]:
            name=ImagesName[bestMatchIndx].upper()
            print(name)
            y1,x1,y2,x2= faceLoc
            cv2.rectangle(frames,(x1,y1),(x2,y2),(0,255,0),1)
            cv2.putText(frames,name,(x2-15,y2+15),cv2.FONT_HERSHEY_SIMPLEX,1/2,(255, 255, 0),1)
            markAttendance(name)
    # cv2.rectangle(frames, (faces_Curr_frame[3], faces_Curr_frame[0]), (faces_Curr_frame[1], faces_Curr_frame[2]), (0, 255, 0), 2)
    cv2.imshow("WEBCAM",frames)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
