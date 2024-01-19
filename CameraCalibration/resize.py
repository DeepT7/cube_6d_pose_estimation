import cv2 
import os
import glob
count = 0
images = glob.glob("./images1/*.JPG")
for fname in images:
    img = cv2.imread(fname)
    H, W = img.shape[:2]
    img = cv2.resize(img, (W//4, H//4))
    cv2.imwrite(os.path.join("images2",f'{count}.jpg'), img)
    count +=1 