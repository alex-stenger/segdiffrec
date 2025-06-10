# BEST STUDENT PAPER AWARD ICASSP 2025 (over 3.310 paper accepted) https://2025.ieeeicassp.org/paper-awards/

## Label-constrained unsupervised domain adaptation for semantic segmentation with diffusion models - Alexandre Stenger, Etienne Baudrier, Benoit Naegel, Nicolas Passat - ICASSP 2025

This is the repository for the following paper : https://hal.science/hal-04852579/document

### Training Instruction

This code was runned using ```python-3.11.4```
libraries used can be find in the file ```requirements.txt```

#### Dataset 
The data should be in the following structure :

```
├── train
    ├── img
        ├── images
            ├── 1.png
            ├── 2.png
            ├── ...
    ├── lbl
        ├── labels
            ├── 1.png
            ├── 2.png
            ├── ...
├── val
    ├── img
        ├── images
            ├── 5001.png
            ├── 5002.png
            ├── ...
    ├── lbl
        ├── labels
            ├── 5001.png
            ├── 5002.png
            ├── ...

├── test
    ├── img
        ├── images
            ├── 6001.png
            ├── 6002.png
            ├── ...
    ├── lbl
        ├── labels
            ├── 6001.png
            ├── 6002.png
            ├── ...
```
#### Train

To train the model, you should use the following command 
``` python3 tools/train.py --sample ddim --network unet_attention --run_name current_name --epochs 400 --batch_size 10 --image_size 64 --dataset_source_path path_to_source --dataset_target_path path_to_target --result_path workdir --lambda_st 1e-3 ```

#### Generation (Segmentation)

Then, before the test phase, there is a separated generation phase since we work with diffusion models
``` python3 tools/generate.py --batch_size 20 --image_size 64 --weight_path workdir/name/weights.pt --dataset_path path_to_target --result_path res_path --num_sample 10 ```

#### Test

Finally, the test script can be launched
``` python3 tools/test.py --pred_path path_to_pred --gt_path path_to_gt --threshold_factor 0.5 ```
