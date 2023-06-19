# SHREC2023

## Installation

```
# It is recommended to create a new environment
conda create -n shrec2023 python==3.8
conda activate shrec2023

# [Optional] If you are using CUDA 11.0 or newer, please install `torch==1.7.1+cu110`
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# Install packages and other dependencies
pip install -r requirements.txt
pip install "laspy[lazrs,laszip]"
python setup.py build develop 

```

## Data Preparation

## Change3D
The data should be organized as follows:
```
--2016
     |--test
     |--train
     |--val
--2020
     |--test
     |--train
     |--val
--labeled_point_lists
                    |--2016-2020
                               |--test
                               |--train
                               |--val
```                    
 ## Synthetic
The data should be organized as follows:
```
--time_a
       |--test
       |--train
       |--val
--time_b
       |--test
       |--train
       |--val
--labeled_point_lists_syn
                        |--test
                        |--train
                        |--val
```

## Testing

```
#Set the Python path to include the parent folder by running the following command:
export PYTHONPATH=..:$PYTHONPATH

#The test data needs to be placed in the 'test' folder of each dataset.

#Change3D
#Run the command:
python experiments/changedataset/test.py --snapshot best_changedataset.pth.tar

#Synthetic
#Run the command:
python experiments/synthetic/test.py --snapshot best_synthetic.pth.tar

```

## Credits

We utilize the codebase from the open source implementation of the paper "Geometric Transformer for Fast and Robust Point Cloud Registration" by Zheng Qin, Hao Yu, Changjian Wang, Yulan Guo, Yuxing Peng and Kai Xu. The open source implementation can be found on GitHub at https://github.com/qinzheng93/GeoTransformer.
