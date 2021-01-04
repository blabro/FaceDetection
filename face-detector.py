import cv2

# Load pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to work
#img = cv2.imread('kuba2.jpg')
webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()
    # Converting to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # draw rectangle
    for a, b, c, d in face_coordinates:
        cv2.rectangle(frame, (a, b, c, d), (0, 255, 0), 2)
    cv2.imshow('grayscaled_img', frame)

    key = cv2.waitKey(10)
    if key==113:
        break

webcam.release()
print("complition succeeded ")
