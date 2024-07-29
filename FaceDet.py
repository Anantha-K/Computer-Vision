import cv2 as cv

# Load your image
img = cv.imread('Photos/IMG_1372.jpg')


gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)



haar_cascade = cv.CascadeClassifier('haarcascade.xml')

faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=1) 

for(x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow("Detected faces",img)


cv.waitKey(0)
