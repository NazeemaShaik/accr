import cv2
import time
import numpy as np
import pytesseract
from google.cloud import vision
import io
client = vision.ImageAnnotatorClient.from_service_account_json(filename='C://Users/spenc/Workspace/container/apibot.json')
from tempfile import NamedTemporaryFile
import os
import uuid


# Globals
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
class_names = ['text']
# -----------


# -----------
save_dir = "images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# ------------


# ------------
vc = cv2.VideoCapture("C://Users/spenc/Downloads/side_1.mp4")
# ------------


# -------------
net = cv2.dnn.readNet("yolov4_text_last.weights", "yolov4_text.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)
# --------------


# --------------
def correct_skew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    print(angle)

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return rotated
# -------------


# -------------
#def detect_text(image):
#    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
##    blur = cv2.GaussianBlur(gray, (3, 3), 0)
#
##TODO gray to blur
#    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
#
##    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
##    erosion = cv2.erode(thresh, rect_kern, iterations=2)
#    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#    dilation = cv2.dilate(thresh, rect_kern, iterations=1)
#
#    dilation = cv2.bitwise_not(dilation)
#
#   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#   config = '-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'# --psm 13 --oem 1 --tessdata-dir ./tessdata'
#   text = pytesseract.image_to_string(dilation, lang='eng', config=config)
#
#    return text
# -------------


# ----------------
# ----------------
def detect_text(path):
    """Detects text in the file."""

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    return texts
#-------------------


x = 0
# ------------------
while cv2.waitKey(1) < 1:
    width  = vc.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    width_25 = 0.25 * width
    width_75 = 0.75 * width

    (grabbed, frame) = vc.read()
    if not grabbed:
        exit()
#    x+=1
#    if not x%5==0:
#        continue
#
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

#    for box in boxes:
#        x, y, w, h = box
#        if x > width_25: continue
#        roi = frame[y:y+h, x:x+w]
#        rn = str(uuid.uuid1())
#        save_rn = os.path.join(save_dir, rn) + ".png"
#        cv2.imwrite(save_rn, roi)
#        text = detect_text(save_rn)
#        tobewritten = ""
#        if text:
#            for t in text:
#                tobewritten += t.description
#        save_tbw = os.path.join(save_dir, rn) + ".txt"
#        with open(save_tbw, "w") as handle:
#            handle.writelines(tobewritten)

    for (classid, score, box) in zip(classes, scores, boxes):
#        x, y, w, h = box
#        if x > width_25: continue
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid[0]], score)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    frame_half = cv2.resize(image, (0,0), fx=0.1, fy=0.1)
    cv2.imshow("detections", frame)


