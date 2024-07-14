import cv2
from classifier import train_classifier


def draw_boundary(img, classifier, scalefactor, minNeighbours, color, text, clf):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_image, scalefactor, minNeighbours)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), ((x + w), (y + h)), color, 2)
        id, _ = clf.predict(gray_image[y:y + h, x: x + w])
        if id == 1:
            cv2.putText(img, "Ronit", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords, img


def generate_dataset(img, id, img_id):
    cv2.imwrite("data/user." + str(id) + "." + str(img_id) + ".jpg", img)


def recognize(img, facecascade, clf):
    color = {"blue": (255, 0, 0), "green": (0, 255, 0), "red": (0, 0, 255)}
    coords, img = draw_boundary(img, facecascade, 1.1, 10, color["blue"], "Face", clf)
    return img


def detect(img, facecascade, img_id):
    color = {"blue": (255, 0, 0), "green": (0, 255, 0), "red": (0, 0, 255)}
    coords, img = draw_boundary(img, facecascade, 1.1, 10, color["blue"], "Face")

    if len(coords) == 4:
        roi_img = img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[3]]
        user_id = 2
        generate_dataset(roi_img, user_id, img_id)
    return img


clf = cv2.face.LBPHFaceRecognizer.create()
clf.read("classifier.xml")

facecascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(0)
img_id = 2
while True:
    _, img = video_capture.read()
    #img = recognize(img, facecascade, clf)
    img = detect(img, facecascade, clf)
    cv2.imshow("face detection", img)
    img_id += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
