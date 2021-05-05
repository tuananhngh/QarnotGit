# QarnotGit
### QarnotDataSplit.ipynb
In this notebook, the original data is divided into 3 dataset according to the room they were collected. Each of those 3 datasets is splitted to 8 chunks such that each chunk contains data of 4 weeks. 

Feature "light" is interpolated to get rid of NaN. Some instances are deleted in "babyfoot" and "jacquard" in order to synchronize the  timestamps with babbage, which have some missing measures.

### QarnotScript.py
The training procedure is as follow : 
* Data is fed to model by chunk's orders
* For each chunk :
*   Divide the data by to train and validation by cutoff value (weekly) (function : testcutoff())
*   A standard scaler is applied to train data 
*   The data is then split to couple of (x,y) (function : createXY()
*   An exponential smoothing is applied to (x) (function : toDataLoader())
*   All couple (x,y) is packed into a torch.Dataloader (function : toDataLoader())


