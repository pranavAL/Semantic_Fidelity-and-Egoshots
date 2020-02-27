Egoshots dataset and Semantic Fidelity metric
=====
This repo contains code for our paper "Egoshots, an ego-vision life-logging dataset and semantic fidelity metric to evaluate diversity in image captioning models" accepted at ICLR-2020, MACHINE LEARNING IN REAL LIFE (ML-IRL) workshop.
## Dataset
Egoshots consists of real-life ego-vision images captioned using state of the art image captioning models, and aims at evaluating the robustness, diversity, and sensitivity of these models, as well as providing a real life-logging setting on-the-wild dataset that can aid the task of evaluating real settings. It consists of images from two computer science interns
for 1 month each. Egoshots images are availaible to download at "link" with corresponding captions "link".
## Captioning Egoshots
Unlabelled images of the Egoshots dataset are captioned by exploiting different image captioning models. We limit our work to three models namely:- 
1. [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf)
2. [nocaps: novel object captioning at scale](https://arxiv.org/pdf/1812.08658.pdf)
3. [Decoupled Novel Object Captioner](https://arxiv.org/pdf/1804.03803.pdf)
### Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
    cd ShowAttendAndTell
The images to be captioned are put in the folder `test/images`. The pretrained weigths of the network is extracted from this [link](https://app.box.com/s/xuigzzaqfbpnf76t295h109ey9po5t8p) and extracted in the current folder.

The pre-trained model can be used to caption the dataset by running the following command.
```shell
python main.py --phase=test \
    --model_file='./models/289999.npy' \
    --beam_size=3
```
* All the generated captions are saved in the `test` folder as `results.csv` 
* To caption the Egoshots images and to extract the pre-trained weights most of the codes are built upon this [repository](https://github.com/coldmanck/show-attend-and-tell).
### nocaps: novel object captioning at scale
