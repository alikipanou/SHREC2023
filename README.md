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


## Credits

This project is based on the open source implementation of the paper "Geometric Transformer for Fast and Robust Point Cloud Registration" by Zheng Qin, Hao Yu, Changjian Wang, Yulan Guo, Yuxing Peng and Kai Xu. The open source implementation can be found on GitHub at https://github.com/qinzheng93/GeoTransformer.git.
