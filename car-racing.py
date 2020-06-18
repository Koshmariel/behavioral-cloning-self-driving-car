import numpy as np
import os
import ntpath
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, Dropout, Flatten, Dense
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

datadir = 'Data'
columns = ['center','left', 'right','steering','throttle','reverse','speed']
data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names = columns, index_col=False)
#data.head()

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail

data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)


#check the uniform distribution of left& right turns
num_bins = 25
samples_per_bin = 900
hist, bins = np.histogram(data['steering'], num_bins)
center = (bins[:-1]+bins[1:]) * 0.5 #centering the graph on 0
print(bins)
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
plt.show()

#removing the bias toward steering straight
print('total data:', len(data))
remove_list= []
for j in range(num_bins):
    list_ = []
    for i in range(len(data['steering'])):
        if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
            list_.append(i)
    list_= shuffle(list_)
    list_ = list_[samples_per_bin:]
    remove_list.extend(list_)

print('removed', len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print('remaining data:', len(data))

hist, _ = np.histogram(data['steering'], (num_bins))
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
plt.show()

#print(data.iloc[1])
def load_img_steering(datadir, df):
    image_path = []
    steering = []
    for i in range(len(data)):
        indexed_data= data.iloc[i]
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        image_path.append(os.path.join(datadir, center.strip()))
        steering.append(float(indexed_data[3]))
    image_paths = np.asarray(image_path)
    steerings = np.asarray(steering)
    return image_paths, steerings

image_paths, steerings = load_img_steering(datadir + '/IMG', data)

X_train, X_val, y_train, y_val = train_test_split(image_paths, steerings, test_size=0.2, random_state=0)
print('Training Samples: {}\nValidation Samples: {}'.format(len(X_train), len(X_val)))

fig, axes = plt.subplots(1, 2, figsize=(12,4))
axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
axes[0].set_title('Training set')
axes[1].hist(y_val, bins=num_bins, width=0.05, color='red')
axes[1].set_title('Validation set')
plt.show()

#randomly zoom the image
def zoom(image):
    zoom = iaa.Affine(scale=(1, 1.3))
    image = zoom.augment_image(image)
    return image

image = image_paths[random.randint(0, len(X_train) - 1)]
original_image = mpimg.imread(image)
zoomed_image = zoom(original_image)

fig, axes = plt.subplots(1,2, figsize=(15,10))
fig.tight_layout()
axes[0].imshow(original_image)
axes[0].set_title('Original image')
axes[1].imshow(zoomed_image)
axes[1].set_title('Zoomed image')
plt.show()

#randomly pan the image
def pan(image):
    pan = iaa.Affine(translate_percent= {"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    image = pan.augment_image(image)
    return image

image = image_paths[random.randint(0, len(X_train) - 1)]
original_image = mpimg.imread(image)
panned_image = pan(original_image)

fig, axes = plt.subplots(1,2, figsize=(15,10))
fig.tight_layout()
axes[0].imshow(original_image)
axes[0].set_title('Original image')
axes[1].imshow(panned_image)
axes[1].set_title('Panned image')
plt.show()

#randomly brighten the image
def brighten(image):
    brightness = iaa.Multiply((0.2, 1.2))
    image = brightness.augment_image(image)
    return image

image = image_paths[random.randint(0, len(X_train) - 1)]
original_image = mpimg.imread(image)
brightenned_image = brighten(original_image)

fig, axes = plt.subplots(1,2, figsize=(15,10))
fig.tight_layout()
axes[0].imshow(original_image)
axes[0].set_title('Original image')
axes[1].imshow(brightenned_image)
axes[1].set_title('Brightenned image')
plt.show()

#flip the image
def flip(image, steering_angle):
    image = cv2.flip(image,1)
    steering_angle = -steering_angle #flip the steering_angle
    return image, steering_angle

random_index = random.randint(0, len(X_train) - 1)
image = image_paths[random_index]
steering_angle = steerings[random_index]

original_image = mpimg.imread(image)
flipped_image, flipped_steering_angle = flip(original_image, steering_angle)

fig, axes = plt.subplots(1,2, figsize=(15,10))
fig.tight_layout()
axes[0].imshow(original_image)
axes[0].set_title('Original image, st.angle=' + str(steering_angle))
axes[1].imshow(flipped_image)
axes[1].set_title('Flipped image, st.angle=' + str(flipped_steering_angle))
plt.show()


#prerpocess the images
def img_preprocess(img):
    img = img[60:135,:,:]                      #removing irrelevant parts of the images
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) #color scheme is more efficient at nvidia model
    img = cv2.GaussianBlur(img, (3,3), 0)      #makes smoother image with less noise
    img = cv2.resize(img, (200, 66))
    img = img/255                              #normalization
    return img

#visialize random original & preprocessed images
image = image_paths[random.randint(0, len(X_train) - 1)]
original_image = mpimg.imread(image)
preprocessed_image = img_preprocess(original_image)

fig, axes = plt.subplots(1,2, figsize=(15,10))
fig.tight_layout()
axes[0].imshow(original_image)
axes[0].set_title('Original image')
axes[1].imshow(preprocessed_image)
axes[1].set_title('Preprocessed  image')
plt.show()

#randomly apply augmentations
def random_augment (image, steering_angle):
    image = mpimg.imread(image)
    if np.random.rand() <0.5:
        image = zoom(image)
    if np.random.rand() <0.5:
        image = pan(image)
    if np.random.rand() <0.5:
        image = brighten(image)
    if np.random.rand() <0.5:
        image, steering_angle = flip(image, steering_angle)
    return image, steering_angle

ncol = 2
nrows = 10
fig, axes = plt.subplots(nrows, ncol, figsize=(15,50))
fig.tight_layout()

for i in range(nrows):
    randnum = random.randint(0, len(image_paths) - 1)
    random_image = image_paths[randnum]
    random_steering = steerings[randnum]
    
    original_image = mpimg.imread(random_image)
    augmented_image, steering = random_augment(random_image, random_steering)
    axes[i][0].imshow(original_image)
    axes[i][0].set_title('Original image')
    axes[i][1].imshow(augmented_image)
    axes[i][1].set_title('Augmented image')
    
plt.show()

def batch_generator(image_paths, steering_angle, batch_size, istraining):
    while True:
        batch_img = []
        batch_steering = []
        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            if istraining:
                im, steering = random_augment(image_paths[random_index], steering_angle[random_index])
            else:
                im = mpimg.imread(image_paths[random_index])
                steering = steering_angle[random_index]
            im = img_preprocess(im)
            batch_img.append(im)
            batch_steering.append(steering)
        yield (np.asarray(batch_img), np.asarray(batch_steering))

X_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, True))
X_val_gen, y_val_gen = next(batch_generator(X_val, y_val, 1, False))

fig, axes = plt.subplots(1,2, figsize=(15,10))
fig.tight_layout()
axes[0].imshow(X_train_gen[0])
axes[0].set_title('Training image')
axes[1].imshow(X_val_gen[0])
axes[1].set_title('Validation image')
plt.show()


'''
#preprocess images in the dataset
X_train = np.array(list(map(img_preprocess, X_train)))
X_val = np.array(list(map(img_preprocess, X_val)))

#visialize random preprocessed image from the dataset
plt.imshow(X_train[random.randint(0, len(X_train) - 1)])
plt.title('random preprocessed image from the dataset')
plt.show()
print('X_train shape: ', X_train.shape)
print('X_val shape: ', X_val.shape)
'''

def nvidia_model():
    model = Sequential()
    model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
#    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
#    model.add(Dropout(0.5))
    
    model.add(Dense(50, activation='elu'))
    #model.add(Dropout(0.5))
    model.add(Dense(10, activation='elu'))
#    model.add(Dropout(0.5))
    model.add(Dense(1))
    
    opt = Adam(lr=1e-4)
    model.compile(loss='mse', optimizer=opt)
    return model



model = nvidia_model()

model.summary()


earlystop = EarlyStopping(patience=10,
                          verbose=1,
                          restore_best_weights=True)

#Learning Rate Reduction
#reduce the LR when accucarcy will not increase for 2 steps
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.000001)

callbacks = [earlystop, learning_rate_reduction]


history = model.fit_generator(batch_generator(X_train, y_train, 50, True), 
                              steps_per_epoch = 600,
                              epochs = 20, 
                              validation_data=batch_generator(X_val, y_val, 50, False), 
                              validation_steps = 400,
                              verbose = 1,
                              callbacks=callbacks)
                              #shuffle = 1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epochs')

model.save('model2.h5')