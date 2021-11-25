#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:40:50 2021

@author: eric

Conditioned GAN
condicionada por 8 modos:
    L-J
    V
    S
    D
    festivo o no
    
condicionada por 14 modos:
    dia de la semana
    festivo o no

"""




import random
import numpy as np
from numpy import hstack, zeros, ones
from numpy.random import rand
from numpy.random import randn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
import matplotlib.pyplot as plt


# =============================================================================
#  GAN architecture
# =============================================================================

def define_discriminator(in_shape=(96,1), n_classes=8):
    """Discriminator model

    Args:
        in_shape (tuple, optional): input data dimension. Defaults to (96,1).
        n_classes (int, optional): number of classes/conditions. Defaults to 2
    """

    #* ----------------------- Define architecture
    label_input = layers.Input(shape=(1,))
    # map labels to a 10-element vector
    li_map = layers.Embedding(input_dim=n_classes, output_dim=48)(label_input)

    # pass li_map to a Dense layer.
    # Importantly, the fully connected layer has enough activations that
    # can be reshaped into in_shape
    li_map = layers.Dense(units=96, activation='linear')(li_map)
    li_map = layers.Reshape(target_shape=(96,1))(li_map)

    # concatenated label input with the input pattern ---> 2-channel ts
    ts_input = layers.Input(shape=in_shape)
    merge = layers.Concatenate()([ts_input, li_map])

    # --------- dicriminator
    # down sample
    fe = layers.Conv1D(filters=32, kernel_size=6, strides=2, padding='same')(merge)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    # down sample
    fe = layers.Conv1D(filters=32, kernel_size=6, strides=2, padding='same')(fe)
    fe = layers.LeakyReLU(alpha=0.2)(fe)

    # classifier
    fe = layers.Flatten()(fe)
    # fe = layers.Dropout(0.4)(fe)
    output_layer = layers.Dense(1, activation='sigmoid')(fe)

    #* ----------------------- create model
    model = Model([ts_input, label_input], output_layer)

    #* ----------------------- compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(
        loss='binary_crossentropy',
        optimizer=opt, metrics=['accuracy'])

    return model


def define_generator(latent_dim, n_classes):
    """Generator model

    Args:
        latent_dim (int, optional): latent variable space dimension. Defaults to 10.
        n_classes (int, optional): number of classes/conditions. Defaults to 2.
    """

    #* ----------------------- Define architecture

    label_input = layers.Input(shape=(1,))
    # map label to a 10-element vecto
    li_map = layers.Embedding(input_dim=n_classes, output_dim=48)(label_input)

    # pass li_map to a Dense layer and reshape to generate 24x1 signal
    li_map = layers.Dense(units=12, activation='linear')(li_map)
    li_map = layers.Reshape(target_shape=(12,1))(li_map)

    # Foundation of a 24x1 signal from the latent space vector
    lv_input = layers.Input(shape=(latent_dim,))
    gen = layers.Dense(units=12*32)(lv_input)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.Reshape(target_shape=(12, 32))(gen)

    # merge inputs
    merge = layers.Concatenate()([gen, li_map])

    # --------------- generator

    # upsample to 24 sample
    gen = layers.Conv1DTranspose(
        filters=32, kernel_size=6, strides=2, padding='same')(merge)
    gen = layers.LeakyReLU(alpha=0.2)(gen)

    # upsample to 48X1 sample
    gen = layers.Conv1DTranspose(
        filters=32, kernel_size=6, strides=2, padding='same')(gen)
    gen = layers.LeakyReLU(alpha=0.2)(gen)

    # upsample to 96X1 sample
    gen = layers.Conv1DTranspose(
        filters=32, kernel_size=6, strides=2, padding='same')(gen)
    gen = layers.LeakyReLU(alpha=0.2)(gen)

    # output
    output_layer = layers.Conv1D(
        filters=1, kernel_size=96, activation='tanh', padding='same')(gen)
    #only possitive outputs
    output_layer = layers.ReLU()(output_layer)


    #* ----------------------- create model
    model = Model([lv_input, label_input], output_layer)

    return model


def define_gan(generator, discriminator):
    """GAN model

    Args:
        generator (keras model): generator model
        discriminator (keras model): discriminator model
    """

    discriminator.trainable = False

    #* ----------------------- Define architecture
    # take a point in latent space as input and a class label
    gen_lv, gen_label = generator.input
    # predict whether input was real or fake
    gen_output = generator.output
    #  connect the generated output from the generator and the the class label
    # input, both as input to the discriminator model.
    # This allows the same class label input to flow down into the generator
    # and down into the discriminator.
    gan_output = discriminator([gen_output, gen_label])

    model = Model([gen_lv, gen_label], gan_output)

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(
        loss='binary_crossentropy',
        optimizer=opt, metrics=['accuracy'])

    return model

# =============================================================================
# OTHER FUNCTIONS
# =============================================================================


# generate points in latent space as input for the g_model
def generate_latent_points(latent_dim, n_samples):
    """Generates 'n_samples' points in the latent space for the g_model

    Args:
        latent_dim (int): dimension of the latent space
        n_samples (int): number of samples to be generated
    """
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim,1)
    return x_input

# use the g_model to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
    """Generates n_samples fake examples

    Args:
        g_model (keras model): g_model  model
        latent_dim (int): latent space dimension
        n_samples (int): number of samples to be generated
    """
    # generate points in latent space
    latent_points = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(latent_points)
    # create class labels
    y = zeros((n_samples, 1))
    return X, y

def generate_real_samples(X_train, n_samples):
    """Selects samples from data set.

    Args:
        dataset (2D array): real data set
        n_samples (int): number of samples to be selected
    """
    # cogemos n muestras del train
    indexes = np.random.permutation(len(X_train))[:n_samples]
    
    X = X_train[indexes]
    
    
    # generate class labels
    y = ones((n_samples, 1))

    return X, y

def generate_real_samples_MODOS(X_train, n_samples, df_modos):
    """Selects samples from data set.

    Args:
        dataset (2D array): real data set
        n_samples (int): number of samples to be selected
    """
    # cogemos n muestras del train
    indexes = np.random.permutation(len(X_train))[:n_samples]
    
    X = X_train[indexes]
    modos = np.squeeze(df_modos.iloc[indexes].values)
    
    
    # generate class labels
    y = ones((n_samples, 1))

    return X, y,modos

def generate_fake_samples_MODOS(g_model, latent_dim, n_samples,df_modos):
    """Generates n_samples fake examples

    Args:
        g_model (keras model): g_model  model
        latent_dim (int): latent space dimension
        n_samples (int): number of samples to be generated
    """
    # cogemos n muestras del modo
    indexes = np.random.permutation(len(df_modos))[:n_samples]
    modos = np.squeeze(df_modos.iloc[indexes].values)
    
    # generate points in latent space
    latent_points = generate_latent_points(latent_dim, n_samples)
    
    # predict outputs
    X = g_model.predict([latent_points,modos])
    # create class labels
    y = zeros((n_samples, 1))
    return X, y, modos

def generate_fake_samples_singlemode_test(g_model, latent_dim, n_samples,modo):
    """Generates n_samples fake examples

    Args:
        g_model (keras model): g_model  model
        latent_dim (int): latent space dimension
        n_samples (int): number of samples to be generated
        
    usamos los indexes para plotear muestras en test
    """
    modos = np.ones(n_samples)*modo
    
    # generate points in latent space
    latent_points = generate_latent_points(latent_dim, n_samples)
    
    # predict outputs
    X = g_model.predict([latent_points,modos])

    return X


# =============================================================================
#     START --> load data and GAN definition
# =============================================================================

ATR_colection = [3452, 3460, 3518, 3580, 3581, 3642, 3909, 3910, 3912, 3913, 3917,
       3918, 3919, 3921, 3926, #5467,
       5474, 5476, 5477, 5480, 5481, 5482, 
       5499, 5506, 5530, 5537, 5538, 5539, 5542, 5543, 5544, 5546, 5547, 
       5549, 5555, 5556, 5558, 5562, 5563, 5565, 5568, 5570, 5572, 5576, 
       5578, 5579, 5581, 5583, 5588, 5592, 5600, 5607, 5611, 5616, 5620, 6980] #7085#]

# =============================================================================
# hacemos recursivo
# =============================================================================
for selec in range(len(ATR_colection)-30):
    selec = selec+30
    print('Training a GAN from '+str(ATR_colection[selec]))
    # entrenamos la GAN con los datos de la espira devuelta al comparar el road embedding
    # selec=0
    df_train = pd.read_csv('C-traffic_data_BATCH/'+str(ATR_colection[selec])+'_B.csv', index_col=0)
    
    #cargamos los modos, que son las condiciones de la cGAN
    n_classes=14
    df_modos = pd.read_csv('modos_colombia_'+str(n_classes)+'.csv')
    
    
    n_timesteps = 96
    
    
    #  load train data
    
    dataset = df_train.values
    
    # data normalization
    scaler = MinMaxScaler()
    scaler.fit(dataset)
    X_train = scaler.transform(dataset)
    
    # divisor of 730 samples in dataset
    n_batch = 73
    batch_per_epoch = int(dataset.shape[0]/n_batch)
    half_batch = int(n_batch/2)
    #BIGGER=CLEANER SYNTHETIC SIGNAL
    latent_dim = 96*2
    n_epochs = 150
    
    
    # =============================================================================
    # GAN training
    # =============================================================================
    
    # create the d_model
    d_model = define_discriminator(in_shape=(n_timesteps,1), n_classes=n_classes)
    # create the g_model
    g_model = define_generator(latent_dim, n_classes)
    # create the gan
    gan_model = define_gan(g_model, d_model)
    
    # loss recordings
    d1_loss_hist, d2_loss_hist, g_loss_hist = [], [], []
    d1_acc_hist, d2_acc_hist = [], []
    
    for i in range(n_epochs):
        for j in range(batch_per_epoch):
            # print('Epoch nº:'+str(i))
            # =============================================================================
            # Training with REAL samples  
            # =============================================================================        
            X_real, y_real, modos_real = generate_real_samples_MODOS(X_train,half_batch, df_modos)    
            # update g_model: runs a single gradient update on a single batch of data
            d_loss1, acc1 = d_model.train_on_batch(
                [X_real.reshape(half_batch, n_timesteps, 1),modos_real],
                y_real)
            
            # =============================================================================
            # Training with FAKE samples  
            # =============================================================================
            X_fake, y_fake, modos_fake = generate_fake_samples_MODOS(g_model, latent_dim, half_batch,df_modos)
            # update d_model: runs a single gradient update on a single batch of data
            d_loss2, acc2 = d_model.train_on_batch(
                [X_fake.reshape(half_batch,n_timesteps,1),modos_fake],
                y_fake)
        
            # =============================================================================
            # Training the whole GAN        
            # =============================================================================
            # prepare points in latent space as input for the g_model
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # cogemos n muestras del modo
            indexes = np.random.permutation(len(df_modos))[:n_batch]
            modos = np.squeeze(df_modos.iloc[indexes].values)
        
            # update the g_model via the d_model's error
            g_loss, accg =gan_model.train_on_batch([X_gan,modos], y_gan)
            if j == batch_per_epoch - 1:
                    print('>Epoch %d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                        (i+1, j+1, batch_per_epoch, acc1, acc2, accg))
            # record history
            d1_loss_hist.append(d_loss1)
            d2_loss_hist.append(d_loss2)
            g_loss_hist.append(g_loss)
            d1_acc_hist.append(acc1)
            d2_acc_hist.append(acc2)
    
    # plot loss
    # fig, axs = plt.subplots(2, 1, sharex=True)
    # (ax1,ax2) = axs
    # # binary crossentropy loss
    # ax1.plot(d1_loss_hist, label='Disc_real_loss')
    # ax1.plot(d2_loss_hist, label='Disc_fake_loss')
    # ax1.plot(g_loss_hist, label='GAN_fake_loss')
    # ax1.legend()
    # ax1.set_title('Binary crossentropy loss')
    # ax1.set_ylabel('Loss')
    
    
    # # accuracy of disc and gen
    # ax2.plot(d1_acc_hist, label='Disc_real_acc')
    # ax2.plot(d2_acc_hist, label='Disc_fake_acc')
    # ax2.legend()
    # ax2.set_title('Accuracy of discriminator')
    # ax2.set_ylabel('Acc')
    # ax2.set_xlabel('Epochs')
    
    
    
    # =============================================================================
    # PLOT ZONE test, respecto de los propios datos de entrenamiento
    # =============================================================================
    n_samples_test = 20
    modos_total = np.squeeze(df_modos.values)
    
    tit = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday',
           'Monday (h)','Tuesday(h)','Wednesday(h)','Thursday(h)','Friday(h)','Saturday(h)','Sunday (h)']
    
    fig, ax = plt.subplots(2,7, sharex=True, sharey=True)
    ax = ax.ravel()
    for i in range(n_classes):
        #sacamos los index de samples de ese modo
        filtrado = np.squeeze(np.where(modos_total==i))
        indexes = np.random.permutation(len(filtrado))[:n_samples_test]
        real_test = filtrado[indexes]
        
        x_test = generate_fake_samples_singlemode_test(g_model, latent_dim, n_samples_test,i)
        x_test = scaler.inverse_transform(np.squeeze(x_test))
        for j in range(n_samples_test):        
            ax[i].plot(dataset[real_test[j]],c='r')
            ax[i].plot(x_test[j],c='k')
        
        ax[i].set_title(tit[i])
    
    ax[0].set_ylabel('Flow')
    ax[7].set_ylabel('Flow')
    for i in range(7):
        ax[7+i].set_xlabel('Timestamp')
    
    plt.xlim([0,96])
    fig.suptitle('Synthetic samples by mode (black) and real test samples (red)')
    
    
    #guardamos los modelos generativos ya entrenados
    g_model.save('model_'+str(ATR_colection[selec])+'.h5')



























