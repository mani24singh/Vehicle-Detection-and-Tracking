import cv2
import imutils
import urllib
import numpy as np

cascade_src = 'cars_18.xml'
car_cascade = cv2.CascadeClassifier(cascade_src)

cam = cv2.VideoCapture(0)
url = "https://cdn10.bostonmagazine.com/wp-content/uploads/sites/2/2020/08/cars-covid-fb.jpg"

while True:
    # for the north
    north_detected = 0
    _, north_img = cam.read()  # reading frame from camera
    north_img = imutils.resize(north_img, width=300)  # resize to 300
    north_gray = cv2.cvtColor(north_img, cv2.COLOR_BGR2GRAY)  # color to Grayscale
    north_cars = car_cascade.detectMultiScale(north_gray, 1.1, 1)  # coordinates of vehicle in a frame
    for (nx, ny, nw, nh) in north_cars:
        cv2.rectangle(north_img, (nx, ny), (nx + nw, ny + nh), (0, 0, 255), 2)
    cv2.imshow("North_Feed", north_img)
    b = str(len(north_cars))
    a = int(b)
    north_detected = a
    n = north_detected

    # for the south
    south_detected = 0
    imgPath = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
    south_img = cv2.imdecode(imgNp, -1)
    south_img = imutils.resize(south_img, width=300)  # resize to 300
    south_gray = cv2.cvtColor(south_img, cv2.COLOR_BGR2GRAY)  # color to Grayscale
    south_cars = car_cascade.detectMultiScale(south_gray, 1.1, 1)  # coordinates of vehicle in a frame
    for (sx, sy, sw, sh) in south_cars:
        cv2.rectangle(south_img, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
    cv2.imshow("South_Feed", south_img)
    b1 = str(len(south_cars))
    a1 = int(b1)
    south_detected = a1
    s = south_detected

    # for north print
    print("------------------------------------------------")
    print("North: %d " % (n))
    if n >= 2:
        print("North More Traffic")
    else:
        print("no traffic")

    # for south print
    print("------------------------------------------------")
    print("South: %d " % (s))
    if s >= 2:
        print("South More Traffic")
    else:
        print("no traffic")

    if cv2.waitKey(33) == 27:
        break

cam.release()
cv2.destroyAllWindows()