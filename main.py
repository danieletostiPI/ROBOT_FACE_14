import cv2
import numpy as np
import time
import RPi.GPIO as GP
##############################################
# while loop to get the frames from the camera
cap = cv2.VideoCapture(0)
cap.set(3,320)
cap.set(4,320)



whT = 320   # WIDTH/ HEIGHT OF THE TARGET IMAGE    # 320 per default sett, # 1280x720 width per mac , # 2580x2048 per camera raspberry
confThreshold = 0.5 # if the confidence of each object found is grater than 0.5 then i want to see the object
nmsThreshold = 0.3

xres = 2580
yres = 2048

x = 0
y = 0
w = 0
h = 0

#object list
#/home/danieletostiPI/ROBOT_FACE/
classesFiles = '/home/danieletostiPI/ROBOT_FACE_13/coco.names'    # extract names from coco.names file
classNames = []
with open(classesFiles, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')      # save all the class names in "classNames"

modelConfiguration = '/home/danieletostiPI/ROBOT_FACE_13/yolov3-320.cfg'
modelWeights = '/home/danieletostiPI/ROBOT_FACE_13/yolov3.weights'

# Create network

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#----------------------------------------------------------------------------------------------
GP.setmode(GP.BOARD)
GP.setwarnings(False)

LEDG = 16
BUTTON_STOP = 18

SERVO_ROT = 32

GP.setup(LEDG, GP.OUT)
GP.setup(SERVO_ROT, GP.OUT)

GP.setup(BUTTON_STOP, GP.IN, pull_up_down=GP.PUD_UP)  # Se Button = 0, bottone pigiato.

servo_rot = GP.PWM(SERVO_ROT, 50)  # 50 Hz

#servo_rot.ChangeDutyCycle(0)  # fermo # va da 2.2 a 12.25
#servo_lat.ChangeDutyCycle(0)

DutyCycle1 = 27
dc1 = float(DutyCycle1)

def rotate_right(dc1, inc):
    if dc1 <= 9:
        dc1 = 9
    else:
        dc1 = dc1 - inc
    servo_rot.ChangeDutyCycle(dc1 / 4)
    time.sleep(0.02)
    servo_rot.ChangeDutyCycle(0)
    return dc1


def rotate_left(dc1, inc):
    if dc1 >= 42:
        dc1 = 42
    else:
        dc1 = dc1 + inc
    servo_rot.ChangeDutyCycle(dc1 / 4)
    time.sleep(0.02)
    servo_rot.ChangeDutyCycle(0)
    return dc1
#----------------------------------------------------------------------------------------------
def findObject(outputs, img, objects = []):#, className = []):  ## devi capire come far visualizzare solo il box di persone
    x, y, w, h = 0, 0, 0, 0
    hT, wT, cT = img.shape
    # lists of the object contained in outputs
    bbox = []   # value of x and y and width and height
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            # i want to find the value of the highest probability
            # i want to remove the fist 5 element of the outputs list because i have the value of x and y and width and height that are not objects
            scores = det[5:]
            classId = np.argmax(scores) # index in the array of the maximum value
            confidence = scores[classId] # value of the corresponding index containing the maximum value
            className = classNames[classId]
            if confidence > confThreshold and (className in objects) :
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int(det[0]*wT - w/2), int(det[1]*hT - h/2) # x,y of the center point
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    # riduco il numero di box trovati per uno stesso oggetto prendendo solo l'oggetto con maggiore confidence
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    #print(indices)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
        #cv2.putText(img,f'{classNames[classIds[i]].upper()}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    #print(len(bbox))
    return x, y, w, h
#----------------------------------------------------------------------------------------------
go = True
right = True
left = True
stop = True

spin = 0
inc = 1
enabler = 1
enablel = 1

servo_rot.start(6.8)
#----------------------------------------------------------------------------------------------
# get Image from Internal Camera
while go:
    #--------------------------
    stop = GP.input(BUTTON_STOP)
    GP.output(LEDG, GP.HIGH)
    if not stop:
        go = False
        GP.output(LEDG, GP.LOW)
        print("Well Done")
    # --------------------------
    print("Paletto4")
    success, img = cap.read()
    # convert image in type blob
    blob = cv2.dnn.blobFromImage(img, 1/255,(whT,whT), crop = False)  #[0, 0, 0], 1,
    net.setInput(blob)  # my Input and I have three different output ( output layers )
    print("Paletto5")
    layerNames = net.getLayerNames()    # layernames list
    #net.getUnconnectedOutlayers()   # I have as an output the index of each class names
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]   # outputnames of our list now i have ' yolo_106' etc
    print("Paletto6")
    outputs = net.forward(outputNames)  # now i convert outputs in the object i want to find : output is a list of 3 elements( 3 outputs)
    print("Paletto7")                             # spiegato bene nel video YOLO v3 EASY METHOD | OpenCV Python (2020) p.3 di Murtaza's Workshop
    #print(outputs[0].shape)             # i have the height width xvalue yvalue of the object and the confidence of the object found
    x1, y1, w1, h1 = findObject(outputs, img, objects = ['person'])#, classNames)
    print("Paletto8")
    #className = classNames[classId]
    #if className in objects:
    x = x1
    y = y1
    w = w1
    h = h1
    xcenter = x+w/2
    ycenter = y+h/2
    print(xcenter)
    print(xcenter)
    #cv2.imshow('Image',img)
    cv2.waitKey(1)  # millisecond
