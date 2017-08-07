# Structured Attentions for Visual Question Answering
The repository contains the majority of the code to reproduce the experimental results of the paper **Structured Attentions for Visual Question Answering** on the VQA-1.0 and VQA-2.0 dataset. Currently only the accelerated version of Mean Field is provided, which is used in the VQA 2.0 challenge. 

![framework](https://user-images.githubusercontent.com/18202259/29042302-f15871b0-7be8-11e7-9058-b629ab834f65.png "The framework of the proposed network.")

<div align=center>
The framework of the proposed network. 
</div> 

## Prerequisites
To reproduce the experimental results,
* Clone and compile [mxnet](https://github.com/apache/incubator-mxnet), with **mxnet@c9e252**, **cub@89de7ab**, **dmlc-core@3dfbc6**, **nnvm@d3558d**, **ps-lite@acdb69**, **mshadow@8eb1e0**. There has been some modification on optimizers (and others) in later versions of mxnet, and code in this repository has not been adapted yet.

* ResNet-152 feature of MS COCO images: extracted with [MCB's preprocess code](https://github.com/akirafukui/vqa-mcb). 

* Our training question and answer data for VQA: coming soon.

## Training from scratch

Run `train_VQA.py`

## Pretrained models
Coming soon.

## Citation

If you found this repository helpful, you could cite

```
@article{chen2017sva,
  title={Structured Attentions for Visual Question Answering},
  author={Chen, Zhu and Yanpeng, Zhao and Shuaiyi, Huang and Kewei, Tu and Yi, Ma},
  journal={IEEE International Conference on Computer Vision (ICCV)},
  year={2017},
}
```

## Licence
This code is distributed under MIT LICENSE.


