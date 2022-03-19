# SituFormer
Official implementation of the paper Rethinking the Two-Stage Framework for Grounded Situation Recognition, AAAI 2022.

## Preparation

### Dependencies
Install the dependencies with the following command.
```
pip install -r requirements.txt
```

### Dataset

#### SWiG
Images can be downloaded [here](https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip)
We recommand to symlink the path to the data/. And the path structure should be as follows:

```
├── data
│   ├── global_utils
│   ├── images_512
│   └── SWiG_jsons
```

## Training for Noun model
After the preparation, you can start the training with the following command.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env main_gsr.py --gsr_path data/swig

```

## Citation
Please consider citing our paper if it helps your research.
```
@article{wei2021rethinking,
  title={Rethinking the Two-Stage Framework for Grounded Situation Recognition},
  author={Wei, Meng and Chen, Long and Ji, Wei and Yue, Xiaoyu and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:2112.05375},
  year={2021}
}
```