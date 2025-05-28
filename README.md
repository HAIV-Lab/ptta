# PTTA: Purifying Malicious Samples for Test-Time Model Adaptation
Test-Time Adaptation (TTA) enables deep neural networks to adapt to arbitrary distributions during inference. 
Existing TTA algorithms generally tend to select benign samples that help achieve robust online prediction and stable self-training. 
Although malicious samples that would undermine the model's optimization should be filtered out, it also leads to a waste of test data. 
To alleviate this issue, we focus on how to make full use of the malicious test samples for TTA by transforming them into benign ones, and propose a plug-and-play method, PTTA. 
The core of our solution lies in the purification strategy, which retrieves benign samples having opposite effects on the objective function to perform Mixup with malicious samples, based on a saliency indicator for encoding benign and malicious data. 
This strategy results in effective utilization of the information in malicious samples and an improvement of the models' online test accuracy. 
In this way, we can directly apply the purification loss to existing TTA algorithms without the need to carefully adjust the sample selection threshold. 
Extensive experiments on four types of TTA tasks and classification, segmentation, and adversarial defense demonstrate the effectiveness of our method.

## Installation
In our environment, the requirements are:
- Python == 3.9
- Pytorch == 1.9.0 (built with CUDA 11.1)
- torchvision == 0.10.0 (built with CUDA 11.1)
- timm == 0.4.12

We recommend using a virtual environment. Run:
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.4.12
```

For additional details, refer to the installation guides of [EATA](https://github.com/mr-eggplant/EATA), [DeYO](https://github.com/Jhyun17/DeYO).

## Datasets
- ImageNet: You can download from [ImageNet](https://www.image-net.org/index.php)
- ImageNet-C: You can download from [ImageNet-C](https://zenodo.org/records/2235448#.YpCSLxNBxAc)
- ImageNet-A: You can download from [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)
- ImageNet-R: You can download from [ImageNet-R](https://github.com/hendrycks/imagenet-r)
- ImageNet-V2: You can download from [ImageNet-V2](https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz)
- ImageNet-Sketch: You can download from [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)

You can put all datasets in a directory (e.g., /data/datasets/) and update paths in the commands accordingly.


## Usage

**Single (Continual) Test-Time Adaptation**
For CPL, use:
```
python main_cpl.py --data /data/datasets/ImageNet --data_corruption /data/datasets/ImageNet-C --exp_type ${exp_type} --algorithm ${algorithm} --batch_size 64 --level 5 --loss2_weight ${loss2_weight} --queue_size ${queue_size} --learning_rate 0.001 --neighbor ${neighbor} --network ${network}
```
- exp_type: `each_shift_reset` means Single Test-Time Adaptation, `continual` means Continual Test-Time Adaptation.
- algorithm: `cpl` means use CPL, `ptta_cpl` means use CPL with PTTA.
- loss2_weight: $\alpha$ in equation 8.
- queue_size: the size of Memory Bank.
- neighbor: the top-K samples.
- network: we use `resnet50` (ResNet-50) and `vit` (ViT-B/16).
- for Datasets like ImageNet-A etc., please replace the path in data_corruption.

For Tent ETA EATA DeYO, use:
```
python main.py --data /data/datasets/ImageNet --data_corruption /data/datasets/ImageNet-C --exp_type ${exp_type} --algorithm ${algorithm} --batch_size 64 --level 5 --loss2_weight ${loss2_weight} --queue_size ${queue_size} --learning_rate 0.001 --neighbor ${neighbor} --network ${network}
```
- algorithm: it can be `tent` `ptta_tent` `eta` `ptta_eta` `eata` `ptta_eata` `deyo` `ptta_deyo`.

**Lifelong Test-Time Adaptation**

CPL:
```
python main_lifelong_cpl.py --data /data/datasets/ImageNet/ --data_corruption /data/datasets/ImageNet-C --exp_type 'continual' --algorithm ptta_cpl --batch_size 64 --level 5 --loss2_weight 0.5 --queue_size 1000 --learning_rate 0.001 --neighbor 1 --network vit

```
ETA, EATA, DeYO:
```
python main_lifelong.py --data /data/datasets/ImageNet/ --data_corruption /data/datasets/ImageNet-C --exp_type 'continual' --algorithm ${algorithm} --batch_size 64 --level 5 --loss2_weight 3 --queue_size 1000 --learning_rate 0.001 --neighbor 1 --network vit
```
- algorithm: You can replace it with `ptta_eta` `ptta_eata` `ptta_deyo`.


## Experiments in Table 1
**For Pixel Feature Logit comparison**, replace --algorithm with `ptta_tent_pixel` `ptta_tent_feature` `ptta_eta_pixel` `ptta_eta_feature` `ptta_deyo_pixel` `ptta_deyo_feature`.

**For ID retrieval**:
```
python main_id.py --data /data/datasets/ImageNet/ --data_corruption /data/datasets/ImageNet-C --exp_type 'each_shift_reset' --algorithm ptta_eta --batch_size 64 --level 5 --loss2_weight 1 --queue_size 1000 --learning_rate 0.001 --neighbor 1 --seed 1 --network vit
```

```
python main_cpl_id.py --data /data/datasets/ImageNet/ --data_corruption /data/datasets/ImageNet-C --exp_type 'each_shift_reset' --algorithm ptta_cpl_id --batch_size 64 --level 5 --loss2_weight 1 --queue_size 1000 --learning_rate 0.001 --neighbor 1 --network vit
```

You can refer to the file  [example_shell.sh](example_shell.sh) for some examples.
