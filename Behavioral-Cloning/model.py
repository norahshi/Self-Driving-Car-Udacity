#ssh carnd@54.219.132.173
#scp ./Behavioral-Cloning/upload_AWS/clone.py carnd@52.53.161.246:.
#scp carnd@52.53.161.246:model.h5 ./Behavioral-Cloning/
#source activate carnd-term1
#python drive.py model.h5


import csv 
import cv2 
import numpy as np

#lines=[]
#with open('./data/driving_log.csv') as csvfile:
#    reader=csv.reader(csvfile)
#    for line in reader:
#        lines.append(line)

images=[]
measurements=[]
correction = 0.2 # this is a parameter to tune
TARGET_SIZE = (64,64) #resize image to 64*64
del_rate = 0.7
cut_value = 0.2
skip_count = 0

lines = []  # Contains all the lines in the csv file

with open('./data/driving_log.csv') as csvfile:
    row_count=0
    reader = csv.reader(csvfile)
    for line in reader:
        #while row_count<6:
        lines.append(line)
        #row_count+=1
del lines[0] #delete header line

for line in lines:
    if abs(float(line[3])) < cut_value: # randomly remove 70% of images with <0.1 angle
        if np.random.random() < del_rate:
            skip_count += 1
            continue # continues to next iteration of for loop
                
    for i in range(3):
        source_path = line[i]
        filename=source_path.split('/')[-1]
        current_path='./data/IMG/'+filename
        
        image = cv2.imread(current_path)
        images.append(image)
        
        steering_center = float(line[3]) #measurement
        # create adjusted steering measurements for the side camera images
        if i==0:
            measurement = steering_center
        elif i==1:
            measurement = steering_center + correction
        else:
            measurement = steering_center - correction
        measurements.append(measurement)

#augment data by flipping the image and angle measurement
augmented_images,augmented_measurements = [],[]
for image,measurement in zip (images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

#resize, convert color, and normalzie image data
def process_image(image):
    image = image[70:-25,:,:]
    image = cv2.GaussianBlur(image, (3,3),0)
    image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = image/255.0 - 0.5
    return image
    
processed_augmented_images=[process_image(x) for x in augmented_images]

X_train=np.array(processed_augmented_images)
y_train=np.array(augmented_measurements)



from keras.models import Sequential 
from keras.layers import Flatten, Dense
from keras.layers import Cropping2D
from keras.layers import Convolution2D
from keras.layers import Lambda

model = Sequential()
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu',input_shape=(64,64,3)))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=2)

model.save('model.h5')