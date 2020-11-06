# GP-classification-for-maneuver-detection
The code uses Gaussian binary classification model to detect if the two tracks belong to the same orbit or not.

The Gaussian Process toolbox based on MATLAB is the "gpml-3.1" by Carl Edward Rasmussen & Hannes Nickisch. This can be downloaded from: http://www.gaussianprocess.org/gpml/code/matlab/doc/

The database is simulated by Orekit (https://www.orekit.org/) in Java. The code is accomplished by Dr. Hao Peng.   
Some databases with different maneuvers are saved in "DataCollectionFile-maneuver_database". The training and testing data are collected here.
The users can define and simulate new maneuver data in "OrekitSimulation/Java/src/main/java/co_paper_spaceops/Work_01_all_simulations.java".

*"CollectData.m"*, *"BuildGP.m"* and *"Predict.m"* shall be kept in the **same folder** as *"DataCollectionFile"* in order to make the code run directly.   
Users can also set their data path and gpml path.

## Running the code
### 1. COllect the Data
First run "CollectData.m" to pre-process the data. There are three test data in this code:   
*Test-1* is from the same orbits as training data, with known maneuvers;  
*Test-2* is from different orbits but the same database as training data, with known maneuvers;  
*Test-3* is from the unknown maneuver database.

In the example code, we define the training code is from orbit 1 to orbit 5 with maneuver magnitudes as 3/5/10 m/s.

### 2. Build the GP model
After pre-processing the data, run "BuildGP.m" to train the model. The trained model and other information are saved at "GP_model.mat".  
Note the training step size as **10** in the example code for quick training. For higher accuracy, increase "trainingsteps".

### 3. Prediction Results
Finally run "Predict.m" to test the results with the trained model. The final result is saved at "PredictedResult.mat".
In this example we define the judgement of the valid as **{-0.5, 0.5}**. If the predicted output is greater than 0.5, we assign its predicted value as 1; if the predicted output is less than -0.5, we assign its value as 1. The correct predictions are the cases that the predicted values agree with their corresponding labels. 

The accuracy is defined as the ratio of the total number of the correctly classified data over the total test data size. We also calculate the confusion matrix for each Test data. The results of the three Test data are saved in "Variable". 
Judgement boundaries in "lower_bound" and "higher_bound" can be changed at the beginning of the code.

In the example we set the uncertainty boundary of the prediction using the **"3-sigma"** to obtain the lower and higher boundaries.  
To analysis the results, we obtain the four cases:   
The valid predictions and within the boundaries;  
The valid predictions but out of the boundaries;  
The invalid predictions but within the boundaries;  
The invalid predictions and out of the boundaries.  
We collect these information of the three Test data in "Uncertainty".

In most prediction cases, the truth data is bounded in the "3-sigma" boundaries except in some cases.
To look at those conditions the truth point is out of the boundaries, we account the index of the points in variables *"error_1"* and *"error_2"* for Test-1 with its higher boundary and lower boundary, *"error_3"* and *"error_4"* for the corresponding boundaries of Test-2, and *"error_5"* and *"error_6"* for Test-3. 
We plot an example figure to show the zoom-in details, including the valid predictioin and the outof boundary condition in Test-3 and saved as "Test-3：ZoominResult.png".



