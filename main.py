import cv2
import numpy as np
import os

orb = cv2.ORB_create(nfeatures=5000) #define number of features
thres = 50 #defines threshold

path = 'DATABASE/'
images =[] 
classNames =[]

myList = os.listdir(path) #list of all files from the database from which key points will be selected

print('Total Database Files Detected:',len(myList))

for cl in myList: #get classes names based on file name
    imgCur = cv2.imread(f'{path}/{cl}',0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])

def findDes(images): #find descriptors using ORB
    desList=[]
    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        desList.append(des)
    return desList

def findKp(images):#find keypoints using ORB
    kpList=[]
    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        kpList.append(kp)
    return kpList

#### select file to test ####
user_list =os.listdir('TEST/')
print(user_list)
test_file = input("Please enter TEST file with extension: ")
test_file = "TEST/"+test_file
#### end ####

desList = findDes(images) #find descriptors of DB
kpList = findKp(images)  #find keypoints of DB


img2 = cv2.imread(test_file,0)
kp2 , des2 = orb.detectAndCompute(img2,None) #find descriptors and keypoints of test image
bf = cv2.BFMatcher() #bruteforce match 
matchList=[]

finalVal = -1
for des in desList:
    matches = bf.knnMatch(des,des2,k=2) # #checking if the match is good based on the distance between the matches (descriptors) If it is greater than the assumed lower limit, then the detected object
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    matchList.append(len(good)) #append the best match


if max(matchList) >= thres:
    finalVal = matchList.index(max(matchList)) # index of the database file that best matches the one being tested

if finalVal != -1: #if value was change enter the name of the database file that matches the best 
    cv2.putText(img2,classNames[finalVal],(320,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    matches = bf.knnMatch(desList[finalVal],des2,k=2)
    check=[]
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            check.append([m])
    final_img = cv2.drawMatchesKnn(images[finalVal],kpList[finalVal],img2,kp2,check,None,flags=2)
    final_img = cv2.resize(final_img,(1280,720))
    cv2.imshow('final_img',final_img)
    cv2.waitKey(0)
else: #if a similar file is not in the database (not detected)
    print("ERROR: Can not recognize this image")
    cv2.putText(img2,"Can not recognize this image",(100,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    cv2.imshow('final_img',img2)
    cv2.waitKey(0)