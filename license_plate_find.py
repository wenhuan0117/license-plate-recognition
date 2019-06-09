import cv2
import numpy as np
upper_blue=np.array([110,255,255])
lower_blue=np.array([78,80,80])

img=cv2.imread('cp.jpg')
def find_cp(img):
    img_blur=cv2.GaussianBlur(img,(3,3),0)

    hsv=cv2.cvtColor(img_blur,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,lower_blue,upper_blue)


    res=cv2.bitwise_and(img,img,mask=mask)
##    cv2.imshow('mask',res)

    ##gray=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    ret,binary=cv2.threshold(mask,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    kernal1=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    open1=cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernal1,iterations=2)

    dilate1=cv2.dilate(open1,kernal1,iterations=3)
##    cv2.imshow('dilate1',dilate1)
    contours,hierarchy=cv2.findContours(dilate1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for i,contour in enumerate(contours):
        l=np.max(contour[:,:,0])-np.min(contour[:,:,0])
        w=np.max(contour[:,:,1])-np.min(contour[:,:,1])
        area=cv2.contourArea(contour)
        rect=cv2.boundingRect(contour)
        if rect[2]/rect[3]>=1 and rect[2]/rect[3]<4:
    ##        if (area/(rect[2]*rect[3]))<1.2 and (area/(rect[2]*rect[3]))>0.7:
    ##        print(i,area,(rect[2]*rect[3]))
##            cv2.rectangle(img,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,255,255),2)

            cp=img[rect[1]:(rect[1]+rect[3]),rect[0]:(rect[0]+rect[2])]
##        cv2.imshow('cp',img)
        return cp


def detect_cpnum(cp):
    gray=cv2.cvtColor(cp,cv2.COLOR_BGR2GRAY)
    ret,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)

##    kernal1=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
##    open1=cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernal1,iterations=1)
    cp_shape=np.shape(cp)
    cp_area=cp_shape[0]*cp_shape[1]
    contours,hierarchy=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
##    cv2.drawContours(cp,contours,-1,(0,0,255),3)
    cp_number=list()
    for i,contour in enumerate(contours):

        area=cv2.contourArea(contour)
        rect=cv2.boundingRect(contour)

        if rect[3]/rect[2]>=1 and rect[3]/rect[2]<4 and rect[3]*rect[2]>0.03*cp_area and rect[3]*rect[2]<0.5*cp_area:
            cp_number.append(cp[(rect[1]):(rect[1]+rect[3]),(rect[0]):(rect[0]+rect[2])])
##            cv2.rectangle(cp,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,255,255),2)
##    cv2.imshow('cp_number',cp)
    return cp_number



if __name__=='__main__':
    cp=find_cp(img)
    cp_number=detect_cpnum(cp)



