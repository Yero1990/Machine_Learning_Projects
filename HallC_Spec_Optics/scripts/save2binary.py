'''
This script does the following:
1. read the histogram objects for each (Q1,Q2,Q3) optics tune,
2. convert the histogram object to a 2d numpy array (which can be plotted as well)
3. saves the 2d numpy arrays, and corresponding labels to a binary format (.h5)

This binary format can be interpreted by the Convolutional Neural Network (CNNs)
similar to how the MNIST hand-written digits are stored in the database

Code Usage: from the command-line, type:
python save2binary.py <arg>   # <arg> = train, or <arg> = test
to either save to a binary file the training or testing images
to be used by the keras code.
'''
import ROOT
from ROOT import *
import sys
import root_numpy   #(make sure to do: pip install root_numpy, before importig this module)
import h5py         # module to help load/save data in binary format .h5
import matplotlib.pyplot as plt
import numpy as np
import os.path
import codecs  # this module is used to decode binary strings to normal form



from matplotlib import rc
rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"

#-------------
# USER INPUT
#-------------
analysis = sys.argv[1]

    
# Create empty dictionaryarrays to store images, labels and tunes
# (the purpose is to put our exp. data in similar format as MNIST data)
imgDict   = {}
labelDict = {}
titleDict  = {}
tuneDict  = {}

# list of key names for each 2d focal plane plot 
key = ['xfp_vs_yfp', 'xfp_vs_ypfp', 'xfp_vs_xpfp', 'xpfp_vs_yfp', 'xpfp_vs_ypfp', 'ypfp_vs_yfp']

cnt = 0 # (image, Q1, Q2, Q3) image-tune counter

# ---- define min/max stepsize for quad tuning ------

if analysis=='train':
    # training range
    q1_min = 0.90
    q1_max = 1.12
    q1_step = 0.02
    
    q2_min = 0.95
    q2_max = 1.06
    q2_step = 0.01
    
    q3_min = 0.90
    q3_max = 1.12
    q3_step = 0.02

if analysis=='test':
    # testing range
    q1_min = 1.00
    q1_max = 1.0001
    q1_step = 0.01
    
    q2_min = 0.955
    q2_max = 1.065
    q2_step = 0.01
    
    q3_min = 1.00
    q3_max = 1.0001
    q3_step = 0.01



# Loop over each quad tune
for Q1 in np.arange(q1_min, q1_max, q1_step):
    for Q2 in np.arange(q2_min, q2_max, q2_step):
        for Q3 in np.arange(q3_min, q3_max, q3_step):

            
            # define inputfile name
            if analysis=='train':
                ifname = "../ROOTfiles/training_files/shms_pointtarg_7p5deg_2gev_wc_mscat_vac_shms_vary_Q1_%.2f_Q2_%.2f_Q3_%.2f_hist.root" % (Q1, Q2, Q3)
            elif analysis=='test':
                ifname = "../ROOTfiles/test_files/shms_pointtarg_7p5deg_2gev_wc_mscat_vac_shms_vary_Q1_%.2f_Q2_%.3f_Q3_%.2f_hist.root" % (Q1, Q2, Q3)
                

                
            # check if file exists
            if os.path.exists(ifname):
                # Open ROOT file with histogram objects
                tf = TFile(ifname, "READ")
            else:
                continue

 
            # Get histogram object (focal plance correlation plots)
            H_xfp_yfp   = gROOT.FindObject('hxfp_yfp')
            H_xfp_ypfp  = gROOT.FindObject('hxfp_ypfp')
            H_xfp_xpfp  = gROOT.FindObject('hxfp_xpfp')
            H_xpfp_yfp  = gROOT.FindObject('hxpfp_yfp')
            H_xpfp_ypfp = gROOT.FindObject('hxpfp_ypfp')
            H_ypfp_yfp  = gROOT.FindObject('hypfp_yfp')
            
            # convert ROOT histogram to numpy array
            # apply transpose to the 2d matrix, and invert the y-axis (flipud) to recover the orientation of the histogram object
            imgArr_xfp_yfp   = np.flipud((root_numpy.hist2array(H_xfp_yfp)).T)    
            imgArr_xfp_ypfp  = np.flipud((root_numpy.hist2array(H_xfp_ypfp)).T)
            imgArr_xfp_xpfp  = np.flipud((root_numpy.hist2array(H_xfp_xpfp)).T)
            imgArr_xpfp_yfp  = np.flipud((root_numpy.hist2array(H_xpfp_yfp)).T)
            imgArr_xpfp_ypfp = np.flipud((root_numpy.hist2array(H_xpfp_ypfp)).T)
            imgArr_ypfp_yfp  = np.flipud((root_numpy.hist2array(H_ypfp_yfp)).T)

            # append optics image patterns to the image dictionary, for each key: 'focal plane 2d plot'
            imgDict.setdefault(key[0], []).append(imgArr_xfp_yfp)
            imgDict.setdefault(key[1], []).append(imgArr_xfp_ypfp)
            imgDict.setdefault(key[2], []).append(imgArr_xfp_xpfp)
            imgDict.setdefault(key[3], []).append(imgArr_xpfp_yfp)
            imgDict.setdefault(key[4], []).append(imgArr_xpfp_ypfp)
            imgDict.setdefault(key[5], []).append(imgArr_ypfp_yfp)

            # append the counter to the label dictionary for each focal plane plot, to keep track of a particular (Q1, Q2, Q3) tune 
            labelDict.setdefault(key[0],[]).append(cnt)
            labelDict.setdefault(key[1],[]).append(cnt)
            labelDict.setdefault(key[2],[]).append(cnt)
            labelDict.setdefault(key[3],[]).append(cnt)
            labelDict.setdefault(key[4],[]).append(cnt)
            labelDict.setdefault(key[5],[]).append(cnt)

            # append numerical values for each of the quads tunes
            tuneDict.setdefault(key[0],[]).append(np.array([Q1, Q2, Q3]))
            tuneDict.setdefault(key[1],[]).append(np.array([Q1, Q2, Q3]))
            tuneDict.setdefault(key[2],[]).append(np.array([Q1, Q2, Q3]))
            tuneDict.setdefault(key[3],[]).append(np.array([Q1, Q2, Q3]))
            tuneDict.setdefault(key[4],[]).append(np.array([Q1, Q2, Q3]))
            tuneDict.setdefault(key[5],[]).append(np.array([Q1, Q2, Q3]))

            # append title-like description, and specify the (Q1,Q2,Q3) config for each individual 2d focal plane plot
            titleDict.setdefault(key[0],[]).append('$x_{fp}$ vs. $y_{fp}$ \n Q1: %.3f, Q2: %.3f, Q3: %.3f' % (Q1, Q2, Q3))
            titleDict.setdefault(key[1],[]).append('$x_{fp}$ vs. $y^{\prime}_{fp}$ \n Q1: %.3f, Q2: %.3f, Q3: %.3f' % (Q1, Q2, Q3))
            titleDict.setdefault(key[2],[]).append('$x_{fp}$ vs. $x^{\prime}_{fp}$ \n Q1: %.3f, Q2: %.3f, Q3: %.3f' % (Q1, Q2, Q3))
            titleDict.setdefault(key[3],[]).append('$x^{\prime}_{fp}$ vs. $y_{fp}$ \n Q1: %.3f, Q2: %.3f, Q3: %.3f' % (Q1, Q2, Q3))
            titleDict.setdefault(key[4],[]).append('$x^{\prime}_{fp}$ vs. $y^{\prime}_{fp}$ \n Q1: %.3f, Q2: %.3f, Q3: %.3f' % (Q1, Q2, Q3))
            titleDict.setdefault(key[5],[]).append('$y^{\prime}_{fp}$ vs. $y_{fp}$ \n Q1: %.3f, Q2: %.3f, Q3: %.3f' % (Q1, Q2, Q3))

            # increment counter for each distinct (Q1, Q2, Q3) configuration
            cnt = cnt+1

            
# Save data images in binary format
if analysis=='train':
    h5f = h5py.File('optics_training.h5', 'w')
elif analysis=='test':
    h5f = h5py.File('optics_test.h5', 'w')
    
# create groups to store hierarchichal data
h5f.create_group('images')
h5f.create_group('labels')
h5f.create_group('tunes')
h5f.create_group('titles')

# loop over each key element and add the dictionary values to each group
for i in range(len(key)):
    
    h5f['images'].create_dataset(key[i], data=imgDict[key[i]])   
    h5f['labels'].create_dataset(key[i], data=labelDict[key[i]])   
    h5f['tunes'].create_dataset(key[i], data=tuneDict[key[i]])   
    h5f['titles'].create_dataset(key[i], data=titleDict[key[i]])   
    
h5f.close()
    


#-------------------------------
# Read/plot HD5 file
# (For instructional purposes)
#-------------------------------

# read  binary data images into array
# This is for instructional purposes, so students
#can see how to read each output from a binary .h5 file
if analysis=='train':
    h5f = h5py.File('optics_training.h5', 'r')
    print('---------------------')
    print('print: h5f.keys()')
    print(h5f.keys())
    print('---------------------')

    print('---------------------')
    print('print: h5f[\'images\'].keys()')
    print(h5f['images'].keys())
    print('---------------------')

    print('---------------------')
    print('print: h5f[\'images\'][\'xfp_vs_xpfp\']')
    print(h5f['images']['xfp_vs_xpfp'])
    a,b,c = h5f['images']['xfp_vs_xpfp'].shape
    print(a, 'images; ', b,'x',c,' pixels per image')
    print('---------------------')

    print('---------------------')
    print('print: h5f[\'labels\'].keys()')
    print(h5f['labels'].keys())
    print('---------------------')

    print('---------------------')
    print('print: h5f[\'labels\'][\'xfp_vs_xpfp\']')
    print(h5f['labels']['xfp_vs_xpfp'])
    print('There are ', h5f['labels']['xfp_vs_xpfp'].shape, ' possible distinct image labels (or equivalently, outcomes)')

    print('---------------------')
    print('print: h5f[\'tunes\'].keys()')
    print(h5f['tunes'].keys())
    print('---------------------')

    print('---------------------')
    print('print: h5f[\'tunes\'][\'xfp_vs_xpfp\']')
    print(h5f['tunes']['xfp_vs_xpfp'])
    print('print: h5f[\'tunes\'][\'xfp_vs_xpfp\'][0]')
    print('[Q1,Q2,Q3] Tunes of image [0]= ', h5f['tunes']['xfp_vs_xpfp'][0])
    print('---------------------')

    print('---------------------')
    print('print: h5f[\'titles\'].keys()')
    print(h5f['titles'].keys())
    print('---------------------')

    print('---------------------')
    print('print: h5f[\'titles\'][\'xfp_vs_xpfp\']')
    print(h5f['titles']['xfp_vs_xpfp'])
    print('print: h5f[\'titles\'][\'xfp_vs_xpfp\'][0]')
    print('Plot Title of image [0]: ', h5f['titles']['xfp_vs_xpfp'][0])

    
