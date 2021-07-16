# ConvGRULane

PyTorch implementation of the paper "[Enhancing Robustness of Lane Detection through Dynamic Smoothness]".

Our paper has been accepted by ICAUS2021.

## Introduction
![network](network.PNG "network")
- RESA shifts sliced
feature map recurrently in vertical and horizontal directions
and enables each pixel to gather global information.
- RESA achieves SOTA results on CULane and Tusimple Dataset.

## Get started
1. Clone the RESA repository
    ```
    git clone https://github.com/Cuibaby/ConvGRULane.git
    ```

2. Create a conda virtual environment and activate it (conda is optional)

    ```Shell
    conda create -n lane python=3.8 -y
    conda activate lane
    ```

3. Install dependencies

    ```Shell
    # Install pytorch firstly, the cudatoolkit version should be same in your system. (you can also use pip to install pytorch and torchvision)
    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

    # Or you can install via pip
    pip install torch torchvision

4. Data preparation

    Download [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3). Then extract them to  `$TUSIMPLEROOT`. Create link to `data` directory.
    
    ```Shell
    cd $RESA_ROOT
    mkdir -p data
    ln -s $TUSIMPLEROOT data/tusimple
    ```
## Training

For training, run

```Shell
python train.py 
```

## Testing
For testing, run
```Shell
python test.py 
```



We provide two trained models on Tusimple and our style lane dataset, downloading our best performed model. (Tusimple: [GoogleDrive](https://drive.google.com/file/d/1M1xi82y0RoWUwYYG9LmZHXWSD2D60o0D/view?usp=sharing)/[BaiduDrive(code:s5ii)](https://pan.baidu.com/s/1CgJFrt9OHe-RUNooPpHRGA),
style lane: [GoogleDrive](https://drive.google.com/file/d/1pcqq9lpJ4ixJgFVFndlPe42VgVsjgn0Q/view?usp=sharing)/[BaiduDrive(code:rlwj)](https://pan.baidu.com/s/1ODKAZxpKrZIPXyaNnxcV3g)
)

## Citation

```BibTeX
```
