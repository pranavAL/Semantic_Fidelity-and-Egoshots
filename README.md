Egoshots dataset and Semantic Fidelity metric
=====
This repo contains code for our paper 

Egoshots, an ego-vision life-logging dataset and semantic fidelity metric to evaluate diversity in image captioning models
accepted at the MACHINE LEARNING IN REAL LIFE (ML-IRL) ICLR 2020 workshop.


## Dataset
Egoshots consists of real-life ego-vision images captioned using state of the art image captioning models, and aims at evaluating the robustness, diversity, and sensitivity of these models, as well as providing a real life-logging setting on-the-wild dataset that can aid the task of evaluating real settings. It consists of images from two computer scientists while interning at Philips Research, Netherlands, for one 1 month each. 

Images are taken automatically by the Autoographer wearable camera when events of interest are detected autonomously.

Egoshots Dataset images are availaible at [Egoshots](https://github.com/NataliaDiaz/Egoshots) repo with corresponding (transfer learning pre-trained) captions [here](https://drive.google.com/open?id=1fHt1GLRsIUNdwvovSINU_CqLMRT6ZTl4).


## Captioning Egoshots

Unlabelled images of the Egoshots dataset are captioned by exploiting different image captioning models. We limit our work to three models, namely:

1. [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf)
2. [nocaps: novel object captioning at scale](https://arxiv.org/pdf/1812.08658.pdf)
3. [Decoupled Novel Object Captioner](https://arxiv.org/pdf/1804.03803.pdf)

### Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
    cd image-captioning/ShowAttendAndTell
```shell
conda create -n myenv
conda activate myenv
pip install -r requirements.txt
```
The images to be captioned need to be placed in the folder `test/images/`. The pretrained weights of the image captioning network are extracted from this [link](https://app.box.com/s/xuigzzaqfbpnf76t295h109ey9po5t8p) into the current folder.

The pre-trained image captioning model can be used to caption the dataset by running the following command:

```shell
python main.py --phase=test \
    --model_file='./models/289999.npy' \
    --beam_size=3
```
* All the generated captions are saved in the `test` folder as `results.csv` 
* To caption the Egoshots images and to extract the pre-trained weights the codes are built upon this [repository](https://github.com/coldmanck/show-attend-and-tell).
### nocaps: novel object captioning at scale
    cd image-captioning/nocaps
To prevent version mismatch (such as GPU Tensorflow version and Caffe conflict) a separate virtual environment is used:
```shell
conda create -n caffe
conda activate caffe
pip install -r requirements.txt
```
The images to be captioned are put in the folder `images/`. 
The pre-trained weights are downloaded using
```shell
./download_models.sh
```
Images are captioned by 
```shell
python noc_captioner.py
```
* The generated captions are saved in the `results` folder.
* The code for captioning the images and pre-trained weights are built upon this [repository](https://github.com/vsubhashini/noc).
### Decoupled Novel Object Captioner
    cd image-captioning/dnoc
The images to be captioned are put in the folder `prepare_data\mscoco\val2014\`. All the images are pre-processed using the following command:
```shell
conda activate myenv
cd prepare_data
sh step2_detection.sh
sh step3_image_feature_extraction.sh
sh step4_transfer_coco_to_noc.sh
python run.py
cd ..
```
The pre-trained weights of the model can be downloaded from [here](https://drive.google.com/file/d/1NNUz7FjLDqIzQt0MCb9wnROmlmUzbPRW/view).
The pre-processed images are captioned using
```shell
python run.py --stage test
```
* All generated captions are saved in `dnoc_ego.txt`.
* The code for preparing the data and captioning the images and the pre-trained weights are built upon this [repository](https://github.com/Yu-Wu/Decoupled-Novel-Object-Captioner).
## Object-Detector
To measure the Semantic Fidelity of the given caption all the object classes present in the image also need to be detected. The initial metric uses YOLO-9000 because of its ability to detect 9000 different classes.
### YOLO-9000
    cd image-captioning/YOLO-9000
To detect all the object classes present in each Egoshots images YOLO9000 is used. The detection and the pre-trained weights are extracted using this
[repository](https://github.com/philipperemy/yolo-9000). All images are stored in `darknet/data/EgoShots/`. To run the YOLO-9000 object detector on all the images of the Egoshots dataset, run:
```shell
for i in data/EgoShots/*.jpg; do ./darknet detector test cfg/combine9k.data cfg/yolo9000.cfg ../yolo9000-weights/yolo9000.weights "$i" ; done > detected_object.txt
```
* The detected objects corresponding to each image are stored in the file `detected_object.txt`
## Objects and Caption Annotation
Filter the individual CSV's into a single file for each image their corresponding captions as `Captions.csv` and for each 
image all the object classes as `Objects.csv`.
```shell
python caption_annotation.py
python image-captioning/YOLO-9000/darknet/object_detector_annotation.py
```
## Semantic Fidelity(SF) metric computation
```shell
python metrics.py
```
The code calculates the Semantic Fidelity value for each captions and the final value are saved as `Meta-data.csv`.
### SF and its variants
To check the authenticity of the initial SF metric we compare its similarity with manually annotated images through the use of the Human Semantic Fidelity metric (and various different forms of SFs by comparing the Pearson correlation coef. rho and coef. of determination R^2 values):

The notebook `SFs_plot.ipynb` compares various regression plots for the different variants of SFs with their corresponding confidence interval.
##  Final Caption
The Semantic Fidelity as calculated is used to output the final captions(in the order highest to lowest SF) for the given image as 
```shell
python final_caption.py --image ****.jpg
```
##  Acknowledgement
We thank the work by [Meng-Jiun Chiou](https://github.com/coldmanck/show-attend-and-tell), [vsubhashini](https://github.com/vsubhashini/noc), [Yu-Wu](https://github.com/Yu-Wu/Decoupled-Novel-Object-Captioner) and [Philippe Rémy](https://github.com/philipperemy/yolo-9000) for releasing the pretrained weights of the image captioning and object detector models which helped in labelling the Egoshots dataset. 
