# intruder_detecter_opencv
from tkinter import *
from tkinter.font import families
import glob
import face_recognition
import cv2
import numpy as np
import jovian
import getpass
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import datetime
from mysql_database import connections



def system():
    paths = glob.glob('Images_Data_understanding/*')
    names = []
    images = []
    image_encodings = []
    image_names = []
    count_img = 0
    for i in paths:
        images.append(face_recognition.load_image_file(i))
        image_encodings.append(face_recognition.face_encodings(images[count_img])[0])
        image_names.append(i.split('\\')[-1].split('.')[0])
        count_img+=1
        print(image_names)

    count = 0
    cap = cv2.VideoCapture(0)
    while True:
        ret,frame = cap.read()
        gray = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(gray)
        face_encodings = face_recognition.face_encodings(gray, face_locations)
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(image_encodings, face_encoding)
            name = 'Unknown'
            face_distances = face_recognition.face_distance(image_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                print("known face",image_names[best_match_index])
                name = image_names[best_match_index]
            if(name=='Unknown'):
                print("INTRUDER DETECTED!")
                cv2.imwrite('Intruder/intru-{}.jpg'.format(count),frame)
                count+=1
                print("saving picture")
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.imshow("output",frame)
        if(cv2.waitKey(1)==ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()    
    myPath = glob.glob('EDAI_Group_Project/Edai/Intruder/*')
    global countFolder
    count=0
    for i in myPath:
        img = cv2.imread(i)
        #print(blur)
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray_img, cv2.CV_64F).var()
        if(count % 1 == 0 and blur > 320):
            cv2.imwrite("Intruder/intru-{}.jpg".format(count), img)
            count += 1
system()
