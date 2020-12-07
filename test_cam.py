import cv2
import numpy as np

HAAR_CASCADE_XML_FILE_FACE = "/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"

#GSTREAMER_PIPELINE = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'

#GSTREAMER_PIPELINE = "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1920, height=1080, framerate=30/1, format=NV12 ! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616 ! nvvidconv ! nvegltransform ! nveglglessink -e"

def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def faceDetect():
	#Get Haar cascade XML file
	face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_XML_FILE_FACE)
	
	#Video Capturing class from OpenCV
	vid_cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
	if vid_cap.isOpened():
		cv2.namedWindow("Face Detection Window", cv2.WINDOW_AUTOSIZE)
		
		while True:
			return_key, image = vid_cap.read()
			if not return_key:
				break
				
			grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			#detected_faces = face_cascade.detectMultiScale(grayscale_image, 1.3, 5)
			
			#Create rect around face
			for(x_pos, y_pos, width, height) in detected_faces:
				cv2.rectangle(image, (x_pos, y_pos), (x_pos + width, y_pos + height), (0, 0, 0), 2)
				
			cv2.imshow("Face Detection Window", image)
			
			key = cv2.waitkey(30) & 0xff
			# Stop program on ESC key
			
			if key == 27:
			
				break
				
		vid_cap.release()
		cv2.destroyAllWindows()
	else:
		print("Cannot open Camera")
		
def show_camera():
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            cv2.imshow("CSI Camera", img)
            # This also acts as
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")

faceDetect()
