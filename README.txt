# U-Net for Segmentation of Transmission Electron Microscopy Images

## Description
This project utilizes the TEM-ImageNet-v1.3 dataset, which can be downloaded from the following GitHub repository: https://github.com/xinhuolin/TEM-ImageNet-v1.3. The dataset includes images and circular masks, which are used in the training process of our model.

## Usage
To run the training script train.py, you need to specify the directories of the images and circular masks from the TEM-ImageNet-v1.3 dataset. You can set these directories as arguments for `dir_img` and `dir_mask` in the `train.py` script.

The dir_checkpoint argument in the train.py script specifies the directory where the training checkpoints will be saved.

In the models directory, you can find five different models: unet_2_layer, unet_3_layer, unet_4_layer, unet_cnn, and unet_wo_skip. You can choose a specific model to train by inputting the corresponding number when prompted during the execution of train.py.

Please refer to the environment.yml file for the required package versions for this project.

## References
TEM-ImageNet-v1.3 dataset: https://github.com/xinhuolin/TEM-ImageNet-v1.3


