# GP-classification-for-maneuver-dection
## The code is for paper "Gaussian-Binary Classification for Resident Space Object Maneuver Detection".

The used databases (in folder DataBase) are all simulated by Orekit.
### Before use, change the path accordingly. 


### To prepare the data for the model, run "CollectData" file. 
CollectData_GP_classification_test01.m: The original section. Simple, and useless :)
CollectData_GP_classification_test02.m: Make the train/test data in a random order. But can only be used with one file (one maneuver).
CollectData_GP_classification_test03.m:: There are two test data type.
                                          Test-1 is from the same orbit as the training data. Test-2 is from different orbits. 
                                          It can also includes multiple maneuvers.
CollectData_GP_classification_test07.m:  There are three test type. 
                                          Test-1 is from the same orbit with same maneuvers as the training data. Test-2 is from different orbits but with same maneuvers. Test-3                                           is from total different database. 

### After collecting the data, run the "GP_classifcation_test". 

Similarly as  the low thrust maneuver files. 

LowThrust_CollectData_GP_classification_07.m collect the hybrid data (impuslive and low thrust) it can also adjhust the sigma (standard deviation) of the noise. Though the data needs to be simulated firstly.

PCA and AE are used with the hybrid conditions. AE is written as python 3.7 with Keras. 

