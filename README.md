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
  *   The data is then split by couple of (x,y) (function : createXY()
  *   An exponential smoothing is applied to (x) (function : toDataLoader())
  *   All couples (x,y) are packed into a torch.Dataloader (function : toDataLoader())

Some parameters to be specified :
* chunk_checkpts : Load the last training checkpts. For the first run, any integer value is accepted, continuous_training must be set to False.
* chunk_tosave : save the trained result to the folder according the trained chunk. For example 1th-chunk is trained, then the saved folder will be 1Chunk 
* continuous_training : True or False

Each chunk will have their own training information such that model checkpoint, loss value, model weight, etc ...

### QarnotDataResult.ipynb

Notebook to evaluate trained model. The same procedure in QarnotScript to get the validation data for each chunk.
