# GP-classification-for-maneuver-dection
The code uses Gaussian binary classification model to detect if the two tracks belong to the same orbit or not.

The Gaussian Process toolbox based on MATLAB is from "gpml-3.1" by Carl Edward Rasmussen & Hannes Nickisch and you can download it from: http://www.gaussianprocess.org/gpml/code/matlab/doc/

The database is simulated by Orekit (https://www.orekit.org/) in Java. The code is accomplished by Dr. Hao Peng. 
Here we provide some database with different maneuvers in "DataCollectionFile-maneuver_database".

Please keep the "CollectData.m", "BuildGP.m" and "Predict.m" files in the *same folder* as "DataCollectionFile" to keep the data path workable. 
You can also set your own data path and gpml path.

First run "CollectData" to pre-process the data. There are three test data in this code: 
Test-1 is from the same orbits as training data, with known maneuvers;
Test-2 is from different orbits but the same database as training data, with known maneuvers;
Test-3 is from unknwon database, with unknown maneuvers.

In the example code, we define the training code is from orbit 1 to orbit 5 with maneuver magnitude as 3/5/10 m/s.

After pre-processing the data, run "BuildGP" to training the model. The trained model and other informations are saved at "GP_model.mat".
Here we set the training size as 10 in the example code for a quick training. If you want to have a higer accuracy, you can increase it.

Finally run "Predict" to predict the results with trained model. The final result is saved at "PredictedResult.mat".
In this example we define the judgement of the valid as {-0.5, 0.5}. If the predicted output is greater than 0.5, we assign its predicted value as 1; if the predicted output is less than -0.5, we assign its value as 1. The correct predictions are the cases that the predicted values agree with their corresponding labels. Counting the total number of the correctly classified data the dividing by the total test data size, we calculate the accuracy.
You can also set your own judgement boundaries at the begining of the code.

In the example we set the uncertainty boundary of the prediction follows the "3-sigma" theorem to obtain the lower and higher boundaries.
To analysis the results, we obtain the valid prediction with label -1 in variables "test_i_pred_0_valid", i here is 1/2/3 for Test-1, Test-2 and Test-3. Similarly, the valid prediction with label 1 are obtained in variables "test_i_pred_1_valid".
We also account the number of the cases that is valid but out of the boundaries in variables "test_i_pred_valid_outBoundary" and the invalid case but out of the boundary in variables "test_i_invalid_outBoundary".

In the example code, we plot the overall results of Test-2 and Test-3. You can zoom in for more details.
To look at those conditions the truth point is out of the boundaries, we account the the index of the points in variables "error_1" and "error_2" for Test-1, "error_3" and "error_4" for Test-2, and "error_5" and "error_6" for Test-3. You can refer this with the plot. 
In the example code We plot an invalid prediction but out of the boundary in Test-3 and saved as "Test-3：InvalidPredictionbutoutofBoundary.png".


