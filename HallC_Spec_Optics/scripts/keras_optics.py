'''
Script extracted from blog by Victor Zhou:
Keras for Beginners: Implementing a Convolutional Neural Network

https://victorzhou.com/blog/keras-cnn-tutorial/


I have modified this script and adapted it to analyze the actual JLab Hall C optics data

Code Usage: from the command-line, type:
python keras_optics.py <arg>   # <arg> = train, or <arg> = test
to either train or test the neural network with their corresponding images

'''

import numpy as np
import matplotlib.pyplot as plt
#import mnist
import h5py   # module to load binary data format (.h5)
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
import codecs  # this module is used to decode binary strings to normal form
import sys
# user input, to either traing the neural network, or test it


#PASS USER COMMAND-LINE ARGUMENT TO ANALYSIS
analysis = sys.argv[1]

# create empty lists to append arrays of data
title = []

train_images = []
train_labels = []
train_tunes = []
train_titles = []

test_images = []
test_labels = []
test_tunes = []
test_titles = []

history = []
loss = []
acc = []


    
# Open training data binary data file
f1 = h5py.File('optics_training.h5', 'r')

# loop over each key (e.g. 'xfp_vs_xpfp', etc.)
for i, key in enumerate(f1['images'].keys()):

    print('i = ',i,', key =', key)
    print('sizeof f1[images][ikey] -> ',  f1['images'][key].shape)
    # append all training images/labels/titles/tunes corresponding to each key (i.e., a key is: 'xfp_vs_yfp', or 'xpfp_vs_yfp', etc.) to a list
    train_images.append( f1['images'][key][:] )
    train_labels.append( f1['labels'][key][:] ) 
    train_titles.append( f1['titles'][key][:] ) 
    train_tunes.append(  f1['tunes'][key][:] ) 
    title.append(key)
    
    # normalize train images
    train_images[i] = (train_images[i] / 255) - 0.5
    
    # Reshape the images.
    train_images[i] = np.expand_dims(train_images[i], axis=3)
    
    
# Open testing data binary data file
f2 = h5py.File('optics_test.h5', 'r')
    
# loop over each key
for i, key in enumerate(f2['images'].keys()):

    print('i = ',i,', key =', key)
    
    # append all training images/labels/tunes corresponding to each key (i.e., a key is: 'xfp_vs_yfp', or 'xpfp_vs_yfp', etc.) to a list
    test_images.append( f2['images'][key][:] )
    test_labels.append( f2['labels'][key][:] ) 
    test_tunes.append(  f2['tunes'][key][:] ) 
    test_titles.append(  f2['titles'][key][:] ) 
    
    # normalize test images
    test_images[i] = (test_images[i] / 255) - 0.5
    
    # Reshape the images.
    test_images[i] = np.expand_dims(test_images[i], axis=3)
    
    print('test_images_shape = ', test_images[i].shape) # (60000, 28, 28, 1)
    

#--------------------------
# Building the Model
# (Using Sequential Class)
#--------------------------
'''
Every Keras model is either built using the Sequential class, which represents a 
linear stack of layers, or the functional Model class, which is more customizeable. 
We’ll be using the simpler Sequential model, since our CNN will be a linear stack of layers.
'''


num_filters = 12   #optimum (12 filters)
filter_size = 6    #optimum (6x6 filters)
pool_size   = 6    #optimum (6x6 pool size)

model = Sequential([
    Conv2D(num_filters, filter_size, input_shape=(200, 200, 1)),   #images are 200x200 pixels, the x1 simply means 'one' 200x200 pixelated image'
    MaxPooling2D(pool_size=pool_size),
    Flatten(),
    # 31 possible outcomes, each with a probability where the highest possible outcome is
    # taken to be the 'predicted' outcome by the model
    Dense(31, activation='softmax'), 
])


if analysis == 'train':

    #---------------------
    # Compiling the Model
    #---------------------
    model.compile(
        optimizer="adam",    # adam, RMSprop, SGD, Nadam, Adamax ('adam' is the best optimizer for these studies)
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    color_arr = ['k', 'b', 'g', 'r', 'm', 'violet']
    
    # loop over each of the 6 training sets (set of images for xfp_vs_yfp, xpfp_vs_yfp, etc. . .)
    for i, key in enumerate(f1['images'].keys()):
                            
        #--------------------
        # Training the Model
        #--------------------
        
        ihist = model.fit(
            train_images[i],
            to_categorical(train_labels[i]),
            epochs=100,
            #validation_data=(test_images, to_categorical(test_labels)),
        )
        
        history.append(ihist)
        
        #--------------------
        # Saving the Model
        #--------------------
        
        # save optimized (trained) weights for later use
        model.save_weights('optics_weights_%s.h5' % (key))

        #-------------------------------------
        # Plot the Neural Network Performance
        #-------------------------------------
    
        # Plot the accuracy and loss vs. epochs to determine how well the network has been trained
            
        ith_loss = np.array(history[i].history['loss'])
        ith_acc = np.array(history[i].history['accuracy'])

        loss.append(ith_loss)
        acc.append(ith_acc)

        plt.subplot(121)
        plt.plot(acc[i], linestyle='-',   color=color_arr[i],  label='accuracy: '+title[i])
        plt.title('Model Performance: Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epochs')
        plt.legend()
        
        plt.subplot(122)
        plt.plot(loss[i], linestyle='--', color=color_arr[i],  label='loss: '+title[i])    
        plt.title('Model Performance: Loss')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend()

    plt.show()


elif analysis == 'test':


    #--------------------------------------------------------------------------
    # Open File To Write Results: Accuracy of Model Prediction of Test Images
    #--------------------------------------------------------------------------

    fout = open('CNN_Optics_Summary_Results.txt', 'w')
    fout.write('----------------------\n'
               'Hall C, Jefferson Lab \n'
               '----------------------\n'
               '\n'
               'Suumary of Results for: \n'
               'SHMS Optics Convolutional Neural Network (CNN) Studies\n'
               '\n'
               )
    
    # --------------------------------------------
    # Load the model's saved weights for each key
    # (assumes the weights have already been saved)
    # -------------------------------------------


    # loop over each key type of images (i.e., xfp_vs_yfp, etc. . . )
    for i, key in enumerate(f2['images'].keys()):

        # Define counter for calculating test images accuracy (for each key, counter gets reset)
        valid_cnt = 0  # VALID image counter (counts when the model predicts the input test tune/image)

            
        fout.write('\n'
                   '2D Optics Correlation: %s\n'
                   '---------------------------------\n'
                   % (key) )

        # load the optimized weights for each key
        model.load_weights('optics_weights_%s.h5' % (key))
    
        # Predict labels of all test images per ith key
        predictions = model.predict(test_images[i])
    
        # ---- Print our model's predictions -----
    
        # print predicted label (max output probability corresponding to each of the input test_images)
        # these labels are actually the indices of the training images (e.g., [5, 7, 7, 9, 21, 29] --> indices to access the predicted image/tunes from
        # the training data set, recall the training data set has 31 images per ith key, to for each key, we need to find what is the index [0-30] with
        # max probability which is taken as the predicted value 
        predicted_labels = np.argmax(predictions, axis=1)
        predicted_img    = train_images[i][predicted_labels]  # array of 2d numpy array images predicted by the model (given the input)
        predicted_tunes  = train_tunes[i][predicted_labels]   # array of predicted quad tunes [Q1,Q2,Q3]
        predicted_titles = train_titles[i][predicted_labels]  # array of plot titles corresponding to the predicted images/tunes
        
        true_labels      = np.arange(predicted_labels.size)   # true labels (test images indices in the order they were given as input to the model, e.g., 0, 1, 2 . .)
        true_img         = test_images[i][true_labels]        # array of 2d numpy array of test images given to test the model
        true_tunes       = test_tunes[i][true_labels]
        true_titles      = test_titles[i][true_labels]

        
        #print('%s keras model predictions' % (key))
        #print('============================')
        #print('predicted_labels = ', predicted_labels)
        #print('predicted_images_shape = ', predicted_img.shape)
        #print('predicted_tunes_shape = ', predicted_tunes.shape)
        #print('predicted_titles_shape = ', predicted_titles.shape)
        
        #print('----------------------------')
        #print('true_labels = ', true_labels)
        #print('true_images_shape = ', true_img.shape)
        #print('true_tunes_shape = ', true_tunes.shape)
        #print('true_titles_shape = ', true_titles.shape)
        
        
        fig, ax = plt.subplots(figsize=(12,12))
        plt.subplots_adjust(left=0.01, bottom=0.025, right=0.99, top=0.95, wspace=0, hspace=0.4)

        # loop over all input test images of a specified key
        for idx in range(predicted_labels.size):

        
            fout.write('Input Test Image : %d\n' % (idx+1))
            fout.write('Predicted Tunes [Q1: %.4f, Q2: %.4f, Q3: %.4f]\n'%(predicted_tunes[idx][0], predicted_tunes[idx][1], predicted_tunes[idx][2]))
            fout.write('True Tunes      [Q1: %.4f, Q2: %.4f, Q3: %.4f]\n'%(true_tunes[idx][0], true_tunes[idx][1], true_tunes[idx][2]))
            
            # calculate the difference between the predicted and true [Q1,Q2,Q3] tunes
            # for now, the requirement for a VALID prediction is: a difference of at most 0.005 is allowed for either Q1, Q2, Q3
            diff = np.round(np.abs(predicted_tunes[idx] - true_tunes[idx]), 4)

            fout.write('absolute_diff:      [dQ1: %.4f, dQ2: %.4f, dQ3: %.4f]\n' %(diff[0], diff[1], diff[2]))

            # sum the total difference, dQ1+dQ2+dQ3, to make sure that it sums up to 0.005 (only one of the quads is offset by 0.005)
            total_diff = np.sum(diff)

            # Threshold difference is 0.005 (smallest allowed difference between quad tunes of at most ONLY one quadrupole)
            
            if total_diff <= 0.005:
                #print('GOOD PREDICTION')
                fout.write('---> VALID PREDICTION \n\n')
                pred_label = 'VALID prediction'
                
                # increase valid prediction counter
                valid_cnt = valid_cnt + 1
                
            elif total_diff > 0.005:
                #print('BAD PREDICTION')
                fout.write('---> INVALID PREDICTION \n\n')
                pred_label = 'INVALID prediction'

            
            # ----- General Formula for calculating the pad number on the subplot, based on the index ----
            #  idx   npad_odd = 2*(idx+1) - 1       npad_even = 2*idx + 2
            #  0              1                     2
            #  1              3                     4
            #  2              5                     6
            #  3              7                     8
            
            # define pad numbering per idx to be: (1,2), (3,4), (5,6), . . . etc.  --> (npad_odd, npad_even) --> (predicted, true)
            npad_odd = 2*(idx+1) - 1
            npad_even = (2*idx) + 2
            
            # common title for all plots
            plt.suptitle(title[i])
            
            # left subplot (predicted image/tune)
            #print('predicted tunes = ', predicted_tunes[idx])
            plt.subplot(5, 4, npad_odd) 
            plt.imshow(predicted_img[idx], cmap='gray_r')
            plt.title(codecs.decode(predicted_titles[idx]), fontsize=8)
            plt.plot([], color='k', marker='', label=pred_label)
            plt.legend()
            
            # right subplot (true image/tune)
            #print('true tunes = ', true_tunes[idx])
            plt.subplot(5, 4, npad_even) 
            plt.imshow(true_img[idx], cmap='gray_r')
            plt.title(codecs.decode(true_titles[idx]), fontsize=8)
            plt.plot([], color='k', marker='', label='true')
            plt.legend()
                
                
        # Calculate the Accuracy for each key (i.e, 2D correlation image, xfp_vs_xpfp, etc.)
        test_accuracy = float(valid_cnt) /  predicted_labels.size
            
        fout.write('=======================\n')
        fout.write('ACCURACY: %.2f / %.2f = %.4f  \n' % (float(valid_cnt), predicted_labels.size, test_accuracy ))
        fout.write('=======================\n\n')

        # Save Images
        plt.savefig('final_results_%s.png'%(key)) # change the resolution of the saved image    
        #plt.show()


