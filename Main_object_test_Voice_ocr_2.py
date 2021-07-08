import tkinter as tk
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import numpy as np
import imutils
import time
import cv2
import os
import pyttsx3
import pytesseract



window = tk.Tk()
window.title("object Detector_OCR")

window.geometry('1100x650')
window.configure(background='snow')

message = tk.Label(window, text="Object Detection _ OCR", bg="snow", fg="black", width=48,
                   height=2, font=('times', 30, 'italic bold '))
message.place(x=5, y=10)



# One time initialization
engine = pyttsx3.init()

# Set properties _before_ you add things to say
engine.setProperty('rate', 125)    # Speed percent (can go over 100)
engine.setProperty('volume', 1)  # Volume 0-1


# load the COCO class labels our YOLO model was trained on
LABELS = open("coco.names").read().strip().split("\n")

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def clear():
    cv2.destroyAllWindows()
    rtitle.destroy()
    
    
def analysis():
	global rtitle
	(W, H) = (None, None)
	frame = cv2.imread(path)
	frame = imutils.resize(frame, width=400)
	# if the frame dimensions are empty, grab them
	if W is None or H is None:
	    (H, W) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
 
	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []
	centers = []
	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
 
			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > 0.5:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
 
				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
 
				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
				centers.append((centerX, centerY))
				# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
	texts = []

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
 
			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			#find positions of objects
			centerX, centerY = centers[i][0], centers[i][1]
			if centerX <= W/3:
				W_pos = "left "
			elif centerX <= (W/3 * 2):
				W_pos = "center "
			else:
				W_pos = "right "
						
			if centerY <= H/3:
				H_pos = "top "
			elif centerY <= (H/3 * 2):
				H_pos = "mid "
			else:
				H_pos = "bottom "

			texts.append(H_pos + W_pos + LABELS[classIDs[i]])


	rtitle = tk.Label(text=texts, background = "snow", font=("", 15))
	rtitle.place(x=450, y=550)
	cv2.imshow("Image", frame)
	clearbutton = tk.Button(window, text="Clear", command=clear  ,fg="black"  ,bg="lawn green"  ,width=12  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
	clearbutton.place(x=110,y=500)

	if texts:
	    finaltext = ', '.join(texts)
	    engine.say(finaltext)
	    # Flush the say() queue and play the audio
	    engine.runAndWait()


def analysisocr():
    global rtitle
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'
    tessdata_dir_config = '--tessdata-dir "C:/Program Files (x86)/Tesseract-OCR/tessdata/"'

    text = pytesseract.image_to_string(path,lang='eng',config=tessdata_dir_config)

    print(text)
    
    rtitle = tk.Label(text=text, background = "snow", font=("", 15))
    rtitle.place(x=450, y=550)
    
    clearbutton = tk.Button(window, text="Clear", command=clear  ,fg="black"  ,bg="lawn green"  ,width=12  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
    clearbutton.place(x=110,y=500)

    engine.say(text)
# Flush the say() queue and play the audio
    engine.runAndWait()

def openphoto():
    global path
    path=askopenfilename(filetypes=[("Image File",'.jpg')])
    frame = cv2.imread(path)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cv2image = imutils.resize(cv2image, width=250)
    img = Image.fromarray(cv2image)
    tkimage = ImageTk.PhotoImage(img)
    myvar=tk.Label(window,image = tkimage, height="350", width="350")
    myvar.image = tkimage
    myvar.place(x=390, y=150)
    button2 = tk.Button(window, text="Analyse Image", command=analysis  ,fg="black"  ,bg="lawn green"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
    button2.place(x=800, y=150)
    
def capture():
    global path
    cam = cv2.VideoCapture(0)
    time.sleep(1.0)
    ret, img = cam.read()
    captured = cv2.imwrite("./Captured_images/Captured.jpg", img)
    cam.release()
    path = "./Captured_images/Captured.jpg"
    frame = cv2.imread(path)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cv2image = imutils.resize(cv2image, width=250)
    img = Image.fromarray(cv2image)
    tkimage = ImageTk.PhotoImage(img)
    myvar=tk.Label(window,image = tkimage, height="350", width="350")
    myvar.image = tkimage
    myvar.place(x=390, y=150)
    button2 = tk.Button(window, text="Analyse Image", command=analysis  ,fg="black"  ,bg="lawn green"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
    button2.place(x=800, y=150)
    button3 = tk.Button(window, text=" Analyse_Text",command=analysisocr,fg="white"  ,bg="blue2"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
    button3.place(x=800, y=270)


def openphoto2():
    global path
    path=askopenfilename(filetypes=[("Image File",'')])
    frame = cv2.imread(path)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cv2image = imutils.resize(cv2image, width=250)
    img = Image.fromarray(cv2image)
    tkimage = ImageTk.PhotoImage(img)
    myvar=tk.Label(window,image = tkimage, height="350", width="350")
    myvar.image = tkimage
    myvar.place(x=390, y=150)
##    button2 = tk.Button(text="Analyse Image_text", command=analysisocr)
##    button2.grid(row=5, column=3, padx=10, pady = 10)
    button3 = tk.Button(window, text=" Analyse_Text",command=analysisocr,fg="white"  ,bg="blue2"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
    button3.place(x=800, y=270)
    
    
##button1 = tk.Button(text="Select Photo", command = openphoto)
##button1.grid(row=1, column=2, padx=10, pady = 10)
##
##button5 = tk.Button(text="Select text", command = openphoto2)
##button5.grid(row=1, column=3, padx=10, pady = 10)
##
##capbut = tk.Button(text="Capture", command = capture)
##capbut.grid(row=2, column=2, padx=10, pady = 10)

button1 = tk.Button(window, text="Select photo ",fg="black",command=openphoto ,bg="lawn green"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
button1.place(x=80, y=150)

Button5 = tk.Button(window, text="Select Text",command=openphoto2,fg="white"  ,bg="blue2"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
Button5.place(x=80, y=270)

capbut = tk.Button(window, text="Capture",fg="black",command=capture ,bg="lawn green"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
capbut.place(x=80, y=390)

window.mainloop()
print("[INFO] Closing ALL")
print("[INFO] Closed")
