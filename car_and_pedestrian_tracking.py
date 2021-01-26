import cv2

#video = cv2.VideoCapture('tesla.mp4')
video = cv2.VideoCapture('littleharry.mp4')

img_file = "car6.jpg"

car_tracker_file = 'car_detector.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

#create classifiers
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

while True:

    #read the frame
    read_successful, frame = video.read()

    #safe coding
    if read_successful:
        # converting to grayscale
        black_n_white = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #detect cars and pedestrians
    cars = car_tracker.detectMultiScale(black_n_white)
    pedestrians = pedestrian_tracker.detectMultiScale(black_n_white)

    for a, b, c, d in cars:
        cv2.rectangle(frame, (a, b, c, d), (0, 255, 0), 2)

    for a, b, c, d in pedestrians:
        cv2.rectangle(frame, (a, b, c, d), (0, 255, 255), 2)

    print('cars ', cars)


    #display the image
    cv2.imshow('Window1', frame)

    #wait and do not close the Window1
    cv2.waitKey(1)

print("Code completed")