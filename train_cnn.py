import numpy as np
from smallvggnet import SmallVGGNet
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
#from skimage.transform import resize
import skimage.io as io
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import os

#Data set url: https://www.kaggle.com/ardamavi/sign-language-digits-dataset


#https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/

directory = os.getcwd()+'/Sign-language-digits-dataset/'
#directory2 = os.getcwd()+'/complementary_data/'
x = np.load(directory+'X.npy')
y = np.load(directory+'Y.npy')

digits = y.shape[1] #The total amount of digits encoded
#Get total number of images for each digit
#counts = y.sum(axis=0)
#print("Images count per digit: ", counts)

#Load complementary data
'''
samples_per_symbol = 20
x_c = np.zeros((samples_per_symbol*digits, 64, 64), dtype=np.uint8)
y_c = np.zeros((samples_per_symbol*digits, ), dtype=np.uint8)
for n in range(digits):
    for m in range(samples_per_symbol):
        path = directory2+'sample%d_%d.png' % (n, m)
        x_c[n*samples_per_symbol+m, :, :] = io.imread(path, as_gray=True)
        y_c[n*samples_per_symbol+m] = n
'''
#y_c = np.flip(y, axis=1)

'''
#Resize images (64, 64) -> (32, 32) and flatten them
x_resized = []
for image in x:
    x_resized.append(resize(image, (32, 32)).flatten())

x = np.array(x_resized)
'''

'''
Figures are not stacked in order, a bit of investigation led to discovering the slices of the 'x' matrix 
that correspond to each digit. The 'slices' dictionary contains the interval of samples corresponding to 
each type of digit image.
'''
slices = {0:np.s_[204:409], 1:np.s_[822:1028], \
          2:np.s_[1649:1855], 3:np.s_[1443:1649], \
          4:np.s_[1236:1443], 5:np.s_[1855:], \
          6:np.s_[615:822], 7:np.s_[409:615], \
          8:np.s_[1028:1236], 9:np.s_[:204]}

'''
Reshape the input set 'x' by clustering images with the same annotation.
'''
X = np.concatenate([x[slice_] for slice_ in \
                    list(slices.values())], axis=0)

'''
Create a label matrix for the newly arranged data set
'''
y = np.zeros(X.shape[0], dtype='uint8')
start = 0

for digit in slices.keys():
    size = x[slices[digit]].shape[0]
    y[start:size+start] = digit
    start+=size

#print("label counts", np.unique(y, return_counts=True))

#X = np.concatenate((X, x_c), axis=0)
#y = np.concatenate((y, y_c), axis=0)



'''
Shuffle images and labels
'''
random.seed(42)
indexes = np.arange(X.shape[0], dtype=int)
random.shuffle(indexes)
X = X[indexes]
y = y[indexes]

#Normalize dataset by subtracting the mean to all pixels
#X -= X.mean()

'''
Split training and test sets. Reshape images: (64, 64) => (64, 64, 1)
'''
X_train, X_test, y_train, y_test = train_test_split(np.expand_dims(X, axis=-1), \
                                      y, test_size=0.25, random_state=42)

'''
Encode labels in a one-vs-all fashion
'''
lb = LabelBinarizer()
lb.fit(np.arange(digits, dtype='uint8'))
y_train = lb.transform(y_train)
y_test = lb.transform(y_test)

#Instantiate Data Generator for augmented data set.
datagen = ImageDataGenerator(rotation_range=85, width_shift_range=0.2, \
                    height_shift_range=0.2, zoom_range=0.2, \
                        horizontal_flip=True)



'''
Instantiate model with the images input_shape 64x64x1
'''
model = SmallVGGNet.build(width=64, height=64, depth=1, classes=digits)

'''
Define learning rate, # of epochs, moment and batch_size values
'''
LR = 0.01
EPOCHS = 50
MOMENT = 0.0
BATCH = 32

'''
Instantiate optimizer and Compile model
TO DO: Play with the optimizer params.: lr, decay, momentum
'''
opt = SGD(lr=LR, momentum=MOMENT)  
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]) 

'''
Train model: Use 'model.fit_generator' to train with augmented data generated on the fly. 
Use 'model.fit' to train without augmented data.
'''
train_history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH, validation_data=(X_test, y_test))
'''
train_history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH), steps_per_epoch=len(X_train)//BATCH, \
                    epochs=EPOCHS, validation_data=(X_test, y_test))
'''

'''
Evaluate CNN performance
'''
predictions = model.predict(X_test, batch_size=32)

print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), \
       target_names=[str(x) for x in range(digits)]))


#scores = model.evaluate(X_test, y_test)

#for pair in zip(model.metrics_names, scores):
    #print(" >Test "+pair[0]+": ", pair[1])


'''
Plot performance figs and save a copy on the '/figures' directory
'''
n = np.arange(EPOCHS, dtype="uint8")
fig, ax = plt.subplots(2, 1)
ax[0].plot(n, train_history.history['loss'], label='Train loss')
ax[0].plot(n, train_history.history['accuracy'], label='Train acc.')
ax[0].set_title("History of training loss and accuracy over %d epochs"%EPOCHS)
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss/Accuracy")
ax[0].legend()

ax[1].plot(n, train_history.history['val_loss'], label='Val. loss')
ax[1].plot(n, train_history.history['val_accuracy'], label='Val. acc.')
ax[1].set_title("History of validation loss and accuracy over %d epochs"%EPOCHS)
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss/Accuracy")
ax[1].legend()

plt.subplots_adjust(hspace=0.8)
plt.savefig("figures/metrics_model5.png")
plt.show()

'''
Save entire model
'''
#model.save('models/hand_digits_model5.h5')

'''
Save model's weights
'''
model.save_weights('models/model5_weights.h5')

'''
Save model's architecture on JSON file
'''
with open('models/model5_architecture.json', 'w') as f:
    f.write(model.to_json())