# Impact of Augmentation Settings

Use specCnn model to investigate the effect of varying noise and phase augmentation settings.
There were 1986 seizure datapoints in the training dataset and 1632 in the test dataset.

| ---                      |  ---                    | ---    | ---  | ---     | --- |
| ---                      |  ---                    | ---    | ---  | ---     | --- |

Random Undersampling, no phase augmentation

| noiseAugmentationFactor  |  noiseAugmentationValue |  TPR   | TNR  | Folder  | Notes |
| ----                     |  ----                   | ----   | ---- | ----    | --- |
| 20                       |  50                     | 49     | 96   | specCnn/1 | original version before reducing learning rate to make it train better.|
| ----                     |  ----                   | ----   | ---- | ----    | |
| 20                       |  25                     | 82     | 76 | specCnn/2 | reduced learning rate to get it to train sensibly |
| 20                       |  40                     | 75     | 80   | specCnn/10 |  |
| 20                       |  50                     | 70     | 84   | specCnn/5 |  |
| 20                       |  75                     | 58     | 89 | specCnn/4 |    | 
| 20                       |  100                     | 52     | 91 | specCnn/3 |    | 
| ----                     |  ----                   | ----   | ---- | ----    |
| 10                       |  50                     |   62   | 87 | specCnn/6 |    | 
| 20                       |  50                     | 70     | 84   | specCnn/5 |  |
| 30                       |  50                     |   57   | 91 | specCnn/7 |    | 
| 40                       |  50                     |   91   | 59 | specCnn/8 |    | 
| ----                     |  ----                   | ----   | ---- | ----    | |
| 40                       |  25                     |   70   | 84 | specCnn/9 |    | 
| ----                     |  ----                   | ----   | ---- | ----    | |


Random Oversampling, no phase augmentation

| noiseAugmentationFactor  |  noiseAugmentationValue |  TPR   | TNR  | Folder  | Notes |
| ----                     |  ----                   | ----   | ---- | ----    | --- |
| 20                       |  10                     |  75    |  83  | specCnn/17 |  |
| 20                       |  20                     |  95    |  57  | specCnn/18 |  |
| 20                       |  25                     |  69    |  94  | specCnn/14 | I might have forgotten to delete the run 13 trained model? |
| 20                       |  25                     |  73    |  93  | specCnn/16 |  |
| 20                       |  30                     |  94    |  28  | specCnn/19 | Why is this different do adjacent ones???? |
| 20                       |  30                     |   78   | 83   | specCnn/20 | Repeat of run 19 for comparison |
| 20                       |  40                     |  71    |  93  | specCnn/12 |  |
| 20                       |  75                     |   70   |  94  | specCnn/13 |  |
| 20                       |  100                     |   61   | 93   | specCnn/15 | Validation loss was very good (<0.1) so surprised teh test results were not better |
