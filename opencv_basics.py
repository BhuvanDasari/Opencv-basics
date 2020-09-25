import cv2
import numpy as np
import matplotlib.pyplot as plt

#Reading image
img = cv2.imread("images.jpg")
cv2.imshow("Output", img)
cv2.waitKey(1000)

#Video
cap = cv2.VideoCapture('test.mp4')


while True:
    success,img = cap.read()
    cv2.imshow("output",img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

#webcam
cap= cv2.VideoCapture(0)
cap.set(3,360) #width
cap.set(4,480) #height
cap.set(10,100) #brightness

while True:
    sucess,img = cap.read()
    cv2.imshow("output",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#image modification
kernel = np.ones((5,5),np.uint8)

img = cv2.imread('images.jpg')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7,7),0)
imgcanny = cv2.Canny(img,150,200)
imgDialation = cv2.dilate(imgcanny, kernel,iterations=1)
imgEroded = cv2.erode(imgDialation, kernel, iterations=1)

cv2.imshow('output',img)
cv2.imshow('Gray',imgGray)
cv2.imshow('Blur',imgBlur)
cv2.imshow('canny',imgcanny)
cv2.imshow('dialte',imgDialation)
cv2.imshow('erode',imgEroded)
cv2.waitKey(0)


img = cv2.imread("images.jpg")
print(img.shape)
imgR = cv2.resize(img, (100,100))
print(imgR.shape)
imgC = img[0:100, 0:100]  #cropping
print(imgC.shape)

cv2.imshow("resize", imgR)
cv2.imshow('Cropped', imgC)
cv2.waitKey(0)

#opencv features
img = np.zeros((512,512,3), np.uint8)
img[:] = 0,255,255
cv2.line(img, (0,0), (512,512), (0,0,255), thickness=2)
cv2.rectangle(img, (0,0), (512,512), (0,255,0), thickness=3)
cv2.circle(img, (256,256), 100, (255,0,0), thickness=2)
cv2.putText(img, "hi", (256,256), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255))
cv2.imshow("img", img)

cv2.waitKey(0)

#stacking images
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

img = cv2.imread('Resources/lena.png')
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

imgStack = stackImages(0.5,([img,imgGray,img],[img,img,img]))

# imgHor = np.hstack((img,img))
# imgVer = np.vstack((img,img))
#
# cv2.imshow("Horizontal",imgHor)
# cv2.imshow("Vertical",imgVer)
cv2.imshow("ImageStack",imgStack)

cv2.waitKey(0)

#masking
def empty(a):
    pass
cv2.namedWindow("trackbars")
cv2.resizeWindow('trackbars', 400, 600)
cv2.createTrackbar('hue min', 'trackbars',0, 179, empty)
cv2.createTrackbar('hue max', 'trackbars', 179, 179, empty)
cv2.createTrackbar('sat min', 'trackbars', 48, 255, empty)
cv2.createTrackbar('sat max', 'trackbars', 255, 255, empty)
cv2.createTrackbar('val min', 'trackbars', 224, 255, empty)
cv2.createTrackbar('val max', 'trackbars', 255, 255, empty)

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
while True:

    success, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos('hue min', 'trackbars')
    h_max = cv2.getTrackbarPos('hue max', 'trackbars')
    s_min = cv2.getTrackbarPos('sat min', 'trackbars')
    s_max = cv2.getTrackbarPos('sat max', 'trackbars')
    v_min = cv2.getTrackbarPos('val min', 'trackbars')
    v_max = cv2.getTrackbarPos('val max', 'trackbars')

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)
    imgres = cv2.bitwise_and(img ,img, mask=mask)

    #cv2.imshow('out',img )
    #cv2.imshow('out1',hsv )
    cv2.imshow('mask', mask)
    #cv2.imshow('result', imgres)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#countours and shapes
img = cv2.imread('shapes.png')
imgcountour = cv2.resize(img, (200,200))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (200,200))
img = cv2.GaussianBlur(img, (5,5), 1)
img = cv2.Canny(img, 20, 20)

def get_countour(img):
    countours, heierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in countours:
        area = cv2.contourArea(cnt)
        cv2.drawContours(imgcountour, cnt, -1, (255, 0, 0), 2)
        if area >= 500:
            cv2.drawContours(imgcountour, cnt, -1, (255, 0, 0), 2)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            objcor = len(approx)
            x, y ,w, h = cv2.boundingRect(approx)

            if objcor == 3:objectType = 'tri'
            elif objcor == 4:
                asprRation = float(w/h)
                if asprRation > 0.95 and asprRation< 1.05:objectType = 'square'
                else:objectType ='rect'
            else:objectType = 'None'

            cv2.rectangle(imgcountour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imgcountour, objectType, (x,y), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,0,0), 2)



get_countour(img)
cv2.imshow('img',img)
cv2.imshow('countour', imgcountour)
cv2.waitKey(0)

#face recoginition
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
img = cv2.imread('images.jpg')

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(imgGray, 1.1 ,4)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)

cv2.imshow('face', img)
cv2.waitKey(0)


#wrap prespective
img = cv2.imread("car.jpg")

width,height = 250,350
pts1 = np.float32([[111,219],[287,188],[154,482],[352,440]])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
imgOutput = cv2.warpPerspective(img,matrix,(width,height))


cv2.imshow("Image",img)
cv2.imshow("Output",imgOutput)

cv2.waitKey(0)
