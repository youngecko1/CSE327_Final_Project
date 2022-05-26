import cv2
import numpy as np

drone_classifier = cv2.CascadeClassifier('dronedetector_P4000N10000_WALTAL144.xml')

img = cv2.imread("test_drone_image2.png", cv2.IMREAD_UNCHANGED)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

drone = drone_classifier.detectMultiScale(img_gray, 1.0485258, 6)
print(drone[0][1])

if drone is ():
    print("No drones found")
for (x,y,w,h) in drone:
    cv2.rectangle(img, (x,y), (x+w,y+h), (127,0,255), 2)
    cv2.imshow('Drone Detection', img)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()
