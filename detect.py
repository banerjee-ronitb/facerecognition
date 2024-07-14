import cv2


def draw_boundary_detect(img, classifier, scalefactor, minNeighbours, color):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_image, scalefactor, minNeighbours)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), ((x + w), (y + h)), color, 2)
        coords = [x, y, w, h]
    return coords, img


def generate_dataset(img, id, img_id):
    cv2.imwrite("data/user." + str(id) + "." + str(img_id) + ".jpg", img)


def detect(img, facecascade, img_id, user_id):
    color = {"blue": (255, 0, 0), "green": (0, 255, 0), "red": (0, 0, 255)}
    coords, img = draw_boundary_detect(img, facecascade, 1.1, 10, color["blue"])

    if len(coords) == 4:
        roi_img = img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[3]]
        generate_dataset(roi_img, user_id, img_id)
    return img


facecascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(0)
img_id = 1
user_id = 1
while True:
    _, img = video_capture.read()
    img = detect(img, facecascade, img_id, user_id)
    cv2.imshow("face detection", img)
    img_id += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
