import numpy as np
from extra_lib import test
import cv2

path="./input/1.jpg"
im=cv2.imread(path)
print("----size---",im.shape)
res=np.zeros_like(im)
test.image_enhance_bimef(im,res)
res=np.concatenate([im,res],1)
cv2.imshow("0",res)
cv2.waitKey(0)

cv2.imwrite("./output/bimef_1.jpg",res)

res_0=np.zeros_like(im)
test.image_enhance_lime(im,res_0,0.8)
res_0=np.concatenate([im,res_0],1)
cv2.imshow("1",res_0)
cv2.waitKey(0)

cv2.imwrite("./output/lime_1.jpg",res_0)
