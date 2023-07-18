#!/usr/bin/env python
# coding: utf-8

# # Importing Dependencies

# In[1]:


import soundfile
import numpy as np
import librosa
import glob
import os
import numpy as np
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from tensorflow import keras
from keras import optimizers
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split

#%reload_ext tensorboard


# # Feature Extraction and Loading Dataset

# In[2]:


def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
    return result


# In[3]:


def load_data(train_size=0.8,test_size=0.2):
    X, y = [], []
    try :
        for file in glob.glob("C:/Users/hp/MINI_PROJECT-Speech-Emotion-Recognization/dataset/Actor_*/*.wav"):
            # get the base name of the audio file
            print(file)
            basename = os.path.basename(file)
            print(basename)
          # get the emotion label
            emotion = int2emotion[basename.split("-")[2]]
          # we allow only AVAILABLE_EMOTIONS we set
            if emotion not in AVAILABLE_EMOTIONS:
                continue
          # extract speech features
            features = extract_feature(file, mfcc=True, chroma=True, mel=True)
          # add to data
            X.append(features)
            l={'happy':0.0,'sad':1.0,'neutral':3.0,'angry':4.0}
            y.append(l[emotion])
    except :
         pass
    # split the data to training and testing and return it
    return train_test_split(np.array(X), y, test_size=test_size,train_size=train_size,random_state=7)


# # Defining Emotions

# In[4]:


int2emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}
AVAILABLE_EMOTIONS = {
    "angry",
    "sad",
    "neutral",
    "happy"
}


# In[5]:


X_train, X_test, y_train, y_test = load_data()


# In[6]:


X_train = np.asarray(X_train)
y_train= np.asarray(y_train)
X_test=np.array(X_test)
y_test=np.array(y_test)
X_train.shape,y_train.shape,X_test.shape,y_test.shape
x_traincnn = np.expand_dims(X_train, axis=2)
x_testcnn = np.expand_dims(X_test, axis=2)
x_traincnn.shape,x_testcnn.shape


# # Configuration 1 - CNN_Adam

# In[7]:


import tensorflow as tf
model = Sequential()

model.add(Conv1D(128, 5,padding='same',input_shape=(180,1)))        #1
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))



model.add(Conv1D(128, 5,padding='same',))                           #2
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(8))                                                 #3
model.add(Activation('softmax'))
optimizer=keras.optimizers.Adam(lr=0.001)
model.summary()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


# In[8]:


import datetime
import tensorflow
get_ipython().run_line_magic('reload_ext', 'tensorboard')
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback=tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
get_ipython().run_line_magic('tensorboard', '--logdir logs/fit')


# In[9]:


cnnhistory=model.fit(x_traincnn, y_train, batch_size=20, epochs=50, validation_data=(x_testcnn, y_test),callbacks=[tensorboard_callback])


# In[10]:


em=['happy','sad','neutral','angry']


# # Predicting using Config-1

# In[11]:


predictions = model.predict(x_testcnn)
n=predictions[1]
print(em[1])
print(n)


# In[12]:


loss, acc = model.evaluate(x_testcnn, y_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# In[13]:


filename = "C:/Users/hp/MINI_PROJECT-Speech-Emotion-Recognization/dataset/Actor_02/03-01-01-01-02-01-02.wav"
features = np.array(extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1))
f=np.expand_dims(features,axis=2)
result = model.predict(f)[0]
print("result :",em[int(result[0])])


# # Configuration 2 - CNN_RMSProp

# In[14]:


import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint

from tensorflow import keras
from keras import optimizers


um = Sequential()

um.add(Conv1D(128, 5,padding='same',input_shape=(180,1)))#1
um.add(Activation('relu'))
um.add(Dropout(0.25))
um.add(MaxPooling1D(pool_size=(8)))

um.add(Conv1D(128, 5,padding='same',))                  #2
um.add(Activation('relu'))
um.add(MaxPooling1D(pool_size=(8)))
um.add(Dropout(0.25))

um.add(Conv1D(128, 5,padding='same',))                  #3
um.add(Activation('relu'))
um.add(Dropout(0.25))

um.add(Flatten())
um.add(Dense(8))                                        #4                      
um.add(Activation('softmax'))
opt = keras.optimizers.RMSprop(learning_rate=0.00005,epsilon=None,rho=0.9,decay=0.0)

um.summary()

um.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# In[15]:


umhistory=um.fit(x_traincnn, y_train, batch_size=20, epochs=50, validation_data=(x_testcnn, y_test),callbacks=[tensorboard_callback])


# In[16]:


get_ipython().run_line_magic('reload_ext', 'tensorboard')
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback=tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
get_ipython().run_line_magic('tensorboard', '--logdir logs/fit')


# # Predicting using Config - 2

# In[17]:


loss, acc = um.evaluate(x_testcnn, y_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# # Configuration 2 - CNN2_Adam

# In[18]:


import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from tensorflow import keras
from keras import optimizers


sm = Sequential()

sm.add(Conv1D(128, 5,padding='same',input_shape=(180,1)))#1
sm.add(Activation('relu'))
sm.add(Dropout(0.1))
sm.add(MaxPooling1D(pool_size=(8)))

sm.add(Conv1D(128, 5,padding='same',))                  #2
sm.add(Activation('relu'))
sm.add(MaxPooling1D(pool_size=(8)))
sm.add(Dropout(0.1))

sm.add(Conv1D(128, 5,padding='same',))                  #3
sm.add(Activation('relu'))
sm.add(Dropout(0.1))

sm.add(Conv1D(128, 5,padding='same',))                  #4
sm.add(Activation('relu'))
sm.add(Dropout(0.1))

sm.add(Flatten())
sm.add(Dense(8))                                        #5                     
sm.add(Activation('softmax'))

sm.summary()
sm.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[19]:


smhistory=sm.fit(x_traincnn, y_train, batch_size=20, epochs=50, validation_data=(x_testcnn, y_test),callbacks=[tensorboard_callback])


# In[20]:


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback=tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
get_ipython().run_line_magic('tensorboard', '--logdir logs/fit')


# # Predicting using Config - 3

# In[21]:


loss, acc = sm.evaluate(x_testcnn, y_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# # Configuration 4 - LSTM_RMSProp

# In[22]:


from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras import callbacks


# In[23]:


model_LSTM=Sequential()
model_LSTM.add(layers.LSTM(64,return_sequences=True,input_shape=(180,1)))
model_LSTM.add(layers.LSTM(64))
model_LSTM.add(layers.Dense(8,activation='softmax'))
print(model_LSTM.summary())
model_LSTM.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[24]:


umhistory=model_LSTM.fit(x_traincnn, y_train, batch_size=20, epochs=50, validation_data=(x_testcnn, y_test),callbacks=[tensorboard_callback])


# # Evaluating Config - 4 LSTM

# In[25]:


loss, acc = model_LSTM.evaluate(x_testcnn, y_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# # Performance Analysis and Visualization of all Configurations

# In[26]:


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback=tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
get_ipython().run_line_magic('tensorboard', '--logdir logs/fit')

