# Import the required modules
import cv2, os
import numpy as np
from PIL import Image
from os.path import join
from play_welcome_audio import welcomeAudioPlay
# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
# Path to the training Dataset
training_path = './savingvideoframes'
# For face recognition we will use the the LBPH Face Recognizer 
recognizer = cv2.createLBPHFaceRecognizer()
max_train_images = 100
stream = 0
def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L') 
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(200, 200),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(10)
    # return the images list and labels list
    return images, labels

# capture frames from video
def face_train_video(train_path,subject,max_train,stream):
    cap = cv2.VideoCapture(stream)
    ret=True
    ctr = 0
    # minimum 10 frames/images per video 
    while(ctr < max_train):
        # read till end of frames
        ret, img = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        cv2.imshow("Recognizing Face", img)
        cv2.waitKey(10)
        cv2.imwrite( join(train_path,subject)+ "." + str(ctr) +".jpg",img) # writes image  to disk
        ctr = ctr + 1
    cap.release()
    cv2.destroyAllWindows()

# predict live feed
def face_recognize_video(lab_person_map,stream):
    ret=True
    ctr = 0
    cap = cv2.VideoCapture(stream)
    while(1):
        # ret, img = cap.read()
        ret, img = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        # gray = Image.open(img).convert('L') 
        predict_image = np.array(gray, 'uint8')

        # break
        faces = face_cascade.detectMultiScale(
            predict_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(200,200),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        for (x,y,w,h) in faces:
            nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])   
            print "{} is Recognized with confidence {}".format(lab_person_map[nbr_predicted], conf)
            cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
            cv2.waitKey(2)
        if(conf < 19):
            welcomeAudioPlay(lab_person_map[nbr_predicted])
            break  
    cap.release()
    cv2.destroyAllWindows()

#emptying directory
for f in os.listdir(training_path):
    os.remove(join(training_path,f))
# get feed from video device and store image frames in training folder
inp_subjects = raw_input("Enter number of Subjects ")
subject = ""
lab_person_map=[]
inp_ctr = int(inp_subjects)
for subj_ctr in range(inp_ctr):
    subject = "subject"+str(subj_ctr)
    name1 = raw_input("Enter name for label ")
    lab_person_map.append(name1.lower())
    train1 = raw_input("start training?(Y/N) ")
    if train1.lower() == "y":
        face_train_video(training_path,subject,max_train_images,stream)
    else:
        break


# Call the get_images_and_labels function and get the face images and the 
# corresponding labels
images, labels = get_images_and_labels(training_path)

# Perform the tranining
recognizer.train(images, np.array(labels))


face_cascade = cv2.CascadeClassifier(cascadePath)

# predict on live feed
face_recognize_video(lab_person_map,stream)