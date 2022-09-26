# Omni-DETR: Omni-Supervised Object Detection with Transformers

This is the PyTorch implementation of the [Omni-DETR](https://assets.amazon.science/91/3c/ac87e7dd44789a62e03b2230e0ed/omni-detr-omni-supervised-object-detection-with-transformers.pdf) paper. It is a unified framework to use different types of weak annotations for object detection.

If you use the code/model/results of this repository please cite:
```
@inproceedings{wang2022omni,
  author  = {Pei Wang and Zhaowei Cai and Hao Yang and Gurumurthy Swaminathan and Nuno Vasconcelos and Bernt Schiele and Stefano Soatto},
  title   = {Omni-DETR: Omni-Supervised Object Detection with Transformers},
  booktitle = {CVPR},
  Year  = {2022}
}
```

## Installation

First, install PyTorch and torchvision. We have tested on version of 1.8.1, but the other versions should also be working, e.g. no earlier than 1.5.1.

Our implementation is partially based on [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR/). Please follow its [instruction](https://github.com/fundamentalvision/Deformable-DETR/blob/main/README.md) for other requirements.

## Usage

### Dataset organization

Please organize each dataset as follows,

```
code_root/
└── coco/
  ├── train2017/
  ├── val2017/
  ├── train2014/
  ├── val2014/
  └── annotations/
    ├── instances_train2017.json
    ├── instances_val2017.json
    ├── instances_valminusminival2014.json
    └── instances_train2014.json
└── voc/
  └── VOCdevkit/
    └── VOC2007trainval
      ├── Annotations/
      ├── JPEGImages/
    └── VOC2012trainval/
      ├── Annotations/
      ├── JPEGImages/
    └── VOC2007test/
      ├── Annotations/
      ├── JPEGImages/
    └── VOC20072012trainval/
      ├── Annotations/
      ├── JPEGImages/
 └── objects365/
     ├── train_objects365/
        ├── objects365_v1_00000000.jpg
        ├── ...
     ├── val_objects365/
        ├── objects365_v1_00000016.jpg
        ├── ...
     └── annotations/
        ├── objects365_train.json
        └── objects365_val.json
 └── bees/
     └── ML-Data/
 └── crowdhuman/
    ├── Images/
      |── 273271,1a0d6000b9e1f5b7.jpg
      |── ...
    ├── annotation_train.odgt
    └── annotation_val.odgt
      
```

### Dataset preparation
First go to ``scripts`` folder

```
cd scripts
```

#### COCO
To get the split labeled and omni-labeled datasets
```
python split_dataset_coco_omni.py
```
Add indicator to coco val set
```
python add_indicator_to_coco2017_val.py
```
For experiments compared with UFO, we prepare coco2014 set
```
python add_indicator_to_coco2014.py
```
#### VOC
First need to convert the annotation formats to coco style by
```
python VOC2COCO.py --xml_dir ../voc/VOCdevkit/VOC2007trainval/Annotations --json_file ../voc/VOCdevkit/VOC2007trainval/instances_VOC_trainval2007.json
python VOC2COCO.py --xml_dir ../voc/VOCdevkit/VOC2007test/Annotations --json_file ../voc/VOCdevkit/VOC2007test/instances_VOC_test2007.json
python VOC2COCO.py --xml_dir ../voc/VOCdevkit/VOC2012trainval/Annotations --json_file ../voc/VOCdevkit/VOC2012trainval/instances_VOC_trainval2012.json
```
To combine the annotations of voc07 and voc12 by
```
python combine_voc_trainval20072012.py
```
Add indicator to voc07 and 12
```
python prepare_voc_dataset.py
```
To get the split labeled and omni-labeled datasets
```
python split_dataset_voc_omni.py
```


#### Objects365
First sample a subset from the original whole training set
```
python prepare_objects365_for_omni.py
```
Add indicator to val
```
python add_indicator_to_objects365val.py
```
To get the split labeled and omni-labeled datasets
```
python split_dataset_objects365_omni.py
```

#### Bees
Because the official training set has some broken images (with names from ``Erlen_Erlen_Hive_04_1264.jpg`` to ``Erlen_Erlen_Hive_04_1842.jpg``), we first need to 
manually delete them or run
```
xargs rm -r file_list_to_remove.txt
```
Finally, 3596 samples are kept. Next, convert the annotation formats to coco style by
```
python Bees2COCO.py
```
To split the training and validation set as 8:2
```
python split_bees_train_val.py
```
To get the split labeled and omni-labeled datasets
```
python split_dataset_bees_omni.py
```

#### CrowdHuman
Please follow [repo](https://github.com/xingyizhou/CenterTrack/blob/master/readme/DATA.md) to first convert annotations with odgt format to coco format, or run
```
python convert_crowdhuman_to_coco.py
```
Because we only focus on the full body detection of CrowdHuman, we first extract such annotation by
```
python build_crowdhuman_dataset.py
```
To get the split labeled and omni-labeled datasets
```
python split_dataset_crowdhuman_omni.py
```

### Training Omni-DETR
After preparing datasets, please change the arguments in the config files, such as ``annotation_json_label``, ``annotation_json_unlabel``, according to the name of the generated json file above. The ``BURN_IN_STEP`` argument sometimes also needs to be changed (please refer to our supplementary materials). In our experiments, this hyperparameter does not have a huge impact on the results.

Because semi-supervised learning is just a special case of omni-supervised learning, to generate semi-supervised results, please modify the ratio of ``fully_labeled`` and ``Unsup``, but set others as 0, when splitting the dataset.

Training Omni-DETR on each dataset (from the repo main folder)

#### Training from scratch

```
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/r50_ut_detr_omni_coco.sh
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/r50_ut_detr_omni_voc.sh
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/r50_ut_detr_omni_objects.sh
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/r50_ut_detr_omni_bees.sh
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/r50_ut_detr_omni_crowdhuman.sh
```

#### Training from Deformable DETR
Because our burn-in stage is totally same as Deformable DETR, it is acceptable to start from a Deformable DETR checkpoint to skip the burn-in stage. Just modify the ``resume`` argument in config file above.


Before running the above scripts, you may have to run the below to change access permissions,
```
chmod u+x ./tools/run_dist_launch.sh
chmod u+x ./configs/r50_ut_detr_omni_coco.sh
```

### Training under the setting of COCO35to80
```
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/r50_ut_detr_tagsU_ufo.sh
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/r50_ut_detr_point_ufo.sh
```

### Training under the setting of VOC07to12
```
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/r50_ut_detr_voc07to12_semi.sh
```

### Note
1. Some of our experiments are on 800-pixels images by 8 * GPUs with 32G memory. If such memory is not affordable, please change the argument of ``pixels`` to 600. Then it can work on 8 * GPUs with 16G memory. 
2. This code could have some minor accuracy differences from our paper due to some implementation changes after the paper submission.

## License

This project is under the Apache-2.0 license. See [LICENSE](LICENSE) for details.
