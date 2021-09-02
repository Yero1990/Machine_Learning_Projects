# General Instructions
The general instructions to run the codes from the command line in this directory are as follow: <br>

1. `root -l make_2Doptics.C(<arg>)`, where \<arg\> = `1` (analyze training files) or `2` (analyze test files) <br>
	
	**Description:** This ROOT C++ script reads and loops over ROOTfiles corresponding to specific (Q1,Q2,Q3) settings specified in the actual name of the file being read. Then, for each setting, it reads the relevant leaf variables in the ROOTTree, loops over all entries (events) and fills (per entry) six empty and pre-defined 2D histogram object with the relevant correlated variables. The six 2D correlations are: (xfp\_vs\_yfp), (xfp\_vs\_ypfp), (xfp\_vs\_xpfp), (xpfp\_vs\_yfp), (xpfp\_vs\_ypfp), (ypfp\_vs\_yfp) <br>

	The six correlation plots per each (Q1, Q2, Q3) tunes are saved to a ROOTfile of the same name as the input file, but with the extension *\_hist.root
	
2. `python save2binary.py <arg>`, where \<arg> = `train` or `test`

	**Description:** This python script reads the ROOTfiles (\*_hist.root) containing the 2D histogram objects saved from the previous step (Step 1) and does the following: 
	* for each (Q1, Q2, Q3) tune configuration, read the six 2D histogram objects and convert to numerical array (nxn matrix, where n is the number of pixels or bins along each axis of the 2D histogram)
	
	
	* use each of the six names (e.g., xfp\_vs\_yfp, etc.) as a distinct key element in the following dictionaries: 
		* imgDict =  {}  ---> stores the numerical arrays (or pixelated images)
		* labelDict ={}  ---> stores labels which are numerical identifiers for each of the (Q1,Q2,Q3) tunes,  for example,  label\_0 >> (Q1=0.9, Q2=0.95, Q3=0.9),  label\_1 >> (Q1=0.9, Q2=0.95, Q3=0.92), etc. 		
		* titleDict = {} ----> stores titles to be used when plotting the pixelated images
		* tuneDict  = {} ---> stores numerical 3-element array (Q1, Q2, Q3) (per image for a given key) with the values of the tunes (e.g., [0.9, 0.95, 0.90], [0.9, 0.95, 0.92], etc.) which are to be used for carrying out numerical calculations when comparing the model predictions to the real test images.
	* the dictionary information is saved to a binary file (.h5 ext) in the format of HDF5 (Hierarchical Data Format) designed to store and organize large amounts of data (refer to documentation, [HDF5 for Python] (https://docs.h5py.org/en/latest/index.html)). 	
NOTE: There is an example towards the end of the `save2binary.py`script showing how to read, access and display the information stored in the binary file.

3. `python keras_optics.py <arg>`, where \<arg> = `train` or `test`

	**Description:** This python script reads in the binray files (.h5 ext) with the relevant information discussed in the previous step (Step 2) and either trains or tests the Convolutional Neural Network model built using Keras API.  It is important to note that each of the six 2D correlation images will be trained separately, as each of these correlations has its own set of [Q1,Q2.Q3] tunes corresponding to a different image, which the network needs to learn during the training. Then, during the testing phase, the set of images corresponding to each of the six 2D correlation will be tested separately, and the accuracy of the model (predictive power, or number of images correcly predicted over total images) will be calculated.

	**Steps Taken During Training the Neural Network:**
	* compile the neural network model 
	* loop over each of the six 2D correlations and perform a model fit using the set of training images used for each of these correlations (the number of epochs or iterations over the set of images can also be defined during the fit)	
	* after the model fit (over total number of epochs), append the history information over all epochs to be used for plotting the model performance later on (accuracy and loss versues number of epochs)
	* save each of the six model's optimized weights and biases, as *optics\_weights_key.h5*, where *key* is each of the six configurations (e.g., xfp\_vs\_yfp, xfp\_vs\_xpfp, etc.)
	* plot the accuracy (and loss) versus the number of epochs for each of the six 2D correlations to quantify the performance of the model. A plot will pop-up, which you can re-size and save for later use.

	**Steps Taken During Testing the Neural Network:** 
	
	* compile the neural network model 
	* load the model's optimized weights and biases determined during the training phase (e.g., *optics\_weights_key.h5*)
	* calculate the model's predictions given the input test images (recall, that given an input test image corresponding to one of the six 2D correlations, the model will output a probability for each of the images it was trained with, and the highest probability will be taken as the model's prediction of the test image)
	* extract predicted as well as the test image information (i.e., image matrix, labels, tunes and titles) to compare how well the predicted image matches the input test image (true image)
	* create and save a summary output .txt file summarizing the results of the predicted and test images for each of the 2D correlations, also make the corresponding plots, side-by-side of (predicted, true) images to have a visual of how well they compare >>>  **NOTE:** output summary file is named:  *CNN\_Optics\_Summary\_Results.txt*  and output comparison plots are named: *final\_results\_key.png*, where *key:* is each of the six configurations (e.g., xfp\_vs\_yfp, xfp\_vs\_xpfp, etc.)
	
