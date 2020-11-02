# GP-classification-for-maneuver-dection
The code uses Gaussian binary classification model to detect if the two tracks belong to the same orbit or not.

The Gaussian Process toolbox based on MATLAB is from "gpml-3.1" by Carl Edward Rasmussen & Hannes Nickisch and you can download it from: http://www.gaussianprocess.org/gpml/code/matlab/doc/

The database is simulated by Orekit in Java. Here we provide some database with different maneuvers in "DataCollectionFile-maneuver_database".

First run "CollectData" to pre-process the data. There are three test data in this code: 
Test-1 is from the same orbits as training data, with known maneuvers;
Test-2 is from different orbits but the same database as training data, with known maneuvers;
Test-3 is from unknwon database, with unknown maneuvers.

After pre-processing the data, run "BuildGP" for training and prediction.
