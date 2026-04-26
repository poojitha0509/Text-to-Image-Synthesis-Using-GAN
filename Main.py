
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import simpledialog
from tkinter import filedialog
from keras.models import model_from_json
from random import randrange
from numpy.random import randn
from keras.models import load_model
from matplotlib import pyplot
import pandas as pd
import cv2
import os
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense,Reshape,Flatten,Conv2D,Conv2DTranspose,LeakyReLU,Dropout
from matplotlib import pyplot
import numpy as np
import cv2
from keras.layers import LSTM
from keras.layers import Bidirectional
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pickle
import re
from numpy import dot
from numpy.linalg import norm
import json
from sklearn.feature_extraction.text import TfidfVectorizer


main = tkinter.Tk()
main.title("Text-to-Image Synthesis Using Generative Adversarial Networks") #designing main screen
main.geometry("1300x1200")

global filename
global gan_model
global encoder_model
global X, Y

# function to generate discriminator model
def define_discriminator(in_shape=(32,32,3)):
    model = Sequential()
    # normal
    model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Bidirectional(LSTM(32))) #adding bilstm for text encoding
    # downsample
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# function to generate standalone generator model
def define_generator(latent_dim):
    model = Sequential()
    # foundation for 4x4 image
    n_nodes = 256 * 4 * 4
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 256)))
    # upsample to 8x8
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 16x16
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # output layer
    model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

#load real dataset samples
def load_real_samples():
    XX = np.load('model/Y.npy')
    return XX

# select real samples to generate fake or related images
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return X, y

def upload():
    text.delete('1.0', END)
    global filename
    global X, Y
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    if os.path.exists("model/X.npy"):
        X = np.load("model/X.npy")
        Y = np.load("model/Y.npy")
    else:
        X = []
        Y = []
        desc_file = "raw_2.0.jsonl"
        for line in open(desc_file, 'r'):
            if len(X) < 10000:
                data = json.loads(line)
                fileName = data['filename']
                description = data['description']
                if os.path.exists("Dataset/img_align_celeba/"+fileName):
                    img = cv2.imread("Dataset/img_align_celeba/"+fileName)
                    img = cv2.resize(img, (128,128))
                    data = re.sub('[^A-Za-z]+', ' ',description.lower().strip())
                    X.append(data)
                    Y.append(img)
                    print(data+" "+str(len(X))+" "+str(img.shape))
            else:
                break
        vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=False, norm=None, decode_error='replace')
        X = vectorizer.fit_transform(X).toarray()
        with open('model/vector.txt', 'wb') as file:
            pickle.dump(vectorizer, file)
        file.close()
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X',X)
        np.save('model/Y',Y)
    text.insert(END,"Total images found in dataset : "+str(Y.shape[0])+"\n\n")
    text.insert(END,"Total descriptions found in dataset : "+str(X.shape[0])+"\n\n")
        
def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    print(x_input.shape)
    return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    XX = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return XX, y

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    # save the generator model tile file
    filename = 'model/generator_model_%03d.h5' % (epoch+1)
    g_model.save(filename)

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
            # evaluate the model performance, sometimes
            if (i+1) % 10 == 0:
                summarize_performance(i, g_model, d_model, dataset, latent_dim)    

def ganModel():
    global gan_model
    text.delete('1.0', END)
    if os.path.exists('model/generator_model_001.h5'):
        gan_model = load_model('model/generator_model_001.h5')
        latent_points = generate_latent_points(200, 200)
        X = gan_model.predict(latent_points)
        text.insert(END,'Fully-GAN model generated\n')
        text.insert(END,'GAN generated latent generated points size : '+str(X.shape)+"\n\n")
    else:
        # size of the latent space
        latent_dim = 200
        # create the discriminator
        d_model = define_discriminator()
        # create the generator
        g_model = define_generator(latent_dim)
        # create the gan
        gan_model = define_gan(g_model, d_model)
        # load image data
        dataset = load_real_samples()
        XX = []
        for i in range(len(dataset)):
            img = dataset[i]
            img = cv2.resize(img, (32,32))
            XX.append(img)
        XX = np.asarray(XX)    
        # train model
        train(g_model, d_model, gan_model, XX, latent_dim)

def generateImage(data, features):
    global X, Y
    latent_points = generate_latent_points(200, 200) #making array of 200 to ask GAN to generate 200 images from train model
    predict = gan_model.predict(latent_points) #calling GAN predict model with 200 array size to generate image
    predict = Y
    return predict, X

def textToImage():
    text.delete('1.0', END)
    with open('model/vector.txt', 'rb') as file:
        tfidf = pickle.load(file)
    file.close()
    data = tf1.get()
    data = re.sub('[^A-Za-z]+', ' ',data)
    embed = tfidf.transform([data]).toarray()
    embed = embed.ravel()
    print(embed.shape)
    data = embed[0:1083]
    data = data.reshape(19,19,3)
    data = cv2.resize(data, (32,32))
    generated_image, X = generateImage(data,data)
    max_accuracy = 0
    index = 0
    for i in range(len(X)):
        predict_score = dot(X[i], embed)/(norm(X[i])*norm(embed))
        if predict_score > max_accuracy:
            max_accuracy = predict_score
            index = i
    if max_accuracy >= 0.45:
        text.insert(END,"Input Text: "+tf1.get()+"\n\n")
        text.insert(END,"Prediction Accuracy: "+str(max_accuracy))
        text.update_idletasks()
        predict = generated_image[index]
        predict = cv2.resize(predict,(250,250))
        cv2.imshow("Generated Image",predict)
        cv2.waitKey(0)
    else:
        text.insert(END,"Unable to predict face from given sentence")
 
def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='Text-to-Image Synthesis Using Generative Adversarial Networks')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=520,y=100)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload CelebA Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

ganButton = Button(main, text="Retrieve From Fully Trained GAN Model", command=ganModel)
ganButton.place(x=50,y=150)
ganButton.config(font=font1)

l1 = Label(main, text='Enter Text Here')
l1.config(font=font1)
l1.place(x=50,y=200)

tf1 = Entry(main,width=50)
tf1.config(font=font1)
tf1.place(x=50,y=250)


encoderButton = Button(main, text="Generate Face from Text", command=textToImage)
encoderButton.place(x=50,y=300)
encoderButton.config(font=font1) 

predictButton = Button(main, text="Exit", command=close)
predictButton.place(x=50,y=350)
predictButton.config(font=font1) 


main.config(bg='OliveDrab2')
main.mainloop()
