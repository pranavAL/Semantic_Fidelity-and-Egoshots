Egoshots dataset and Semantic Fidelity metric
=====
This repo contains code for our paper "Egoshots, an ego-vision life-logging dataset and semantic fidelity metric to evaluate diversity in image captioning models" accepted at ICLR-2020, MACHINE LEARNING IN REAL LIFE (ML-IRL) workshop.
## Dataset
Egoshots consists of real-life ego-vision images captioned using state of the art image captioning models, and aims at evaluating the robustness, diversity, and sensitivity of these models, as well as providing a real life-logging setting on-the-wild dataset that can aid the task of evaluating real settings. It consists of images from two computer science interns
for 1 month each. Egoshots images are availaible to download at [link](https://drive.google.com/open?id=1gwg1LhjsqZZpCGJBQihb32E1Y2GCZ-Xr) with corresponding captions from this [link](https://drive.google.com/open?id=1fHt1GLRsIUNdwvovSINU_CqLMRT6ZTl4).
## Captioning Egoshots
Unlabelled images of the Egoshots dataset are captioned by exploiting different image captioning models. We limit our work to three models namely:- 
1. [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf)
2. [nocaps: novel object captioning at scale](https://arxiv.org/pdf/1812.08658.pdf)
3. [Decoupled Novel Object Captioner](https://arxiv.org/pdf/1804.03803.pdf)
### Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
    cd ShowAttendAndTell
```shell
conda create -n myenv
conda activate myenv
pip install -r requirements.txt
```
The images to be captioned are put in the folder `test/images/`. The pretrained weigths of the network is extracted from this [link](https://app.box.com/s/xuigzzaqfbpnf76t295h109ey9po5t8p) and extracted in the current folder.

The pre-trained model can be used to caption the dataset by running the following command.
```shell
python main.py --phase=test \
    --model_file='./models/289999.npy' \
    --beam_size=3
```
* All the generated captions are saved in the `test` folder as `results.csv` 
* To caption the Egoshots images and to extract the pre-trained weights the codes are built upon this [repository](https://github.com/coldmanck/show-attend-and-tell).
### nocaps: novel object captioning at scale
    cd nocaps
To prevent version mismatch or conflicting libraries a separate virtual environment is used. Gpu version of Tensorflow and Caffe conflict.
```shell
conda create -n caffe
conda activate caffe
pip install -r requirements.txt
```
The image to be captioned are put in the folder `images/`. The pre-trained weights are downloaded using
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
    cd dnoc
The images to be captioned are put in the folder `prepare_data\mscoco\val2014\`. All the images are pre-processed using the following command.
```shell
conda activate myenv
cd prepare_data
sh step2_detection.sh
sh step3_image_feature_extraction.sh
sh step4_transfer_coco_to_noc.sh
python run.py
cd ..
```
The pre-trained weights of the model can be downloaded from this [link](https://drive.google.com/file/d/1NNUz7FjLDqIzQt0MCb9wnROmlmUzbPRW/view).
The pre-processed images are captioned using
```shell
python run.py --stage test
```
* All the generated captions are saved in `dnoc_ego.txt`.
* The code for preparing the data and captioning the images and the pre-trained weights are built upon this [repository](https://github.com/Yu-Wu/Decoupled-Novel-Object-Captioner).
## Object-Detector
To measure the Semantic Fidelity of the given caption all the object classes present in the image also needs to be detected. The initial metric use YOLO-9000 because of its ability to detect 9000 different classes.
### YOLO-9000
    cd YOLO-9000
To detect all the object classes present in each Egoshots images YOLO9000 is used. The detection and the pre-trained weights are extracted using this
[repository](https://github.com/philipperemy/yolo-9000). All the images are stored in `darknet/data/EgoShots/`. To run the YOLO-9000 object detector on all the images of the Egoshots dataset the following command is run
```shell
for i in data/EgoShots/*.jpg; do ./darknet detector test cfg/combine9k.data cfg/yolo9000.cfg ../yolo9000-weights/yolo9000.weights "$i" ; done > detected_object.txt
```
* The detected objects corresponding to each image are stored in file `detected_object.txt`
## Objects and Caption Annotation
Filter the individual CSV's into a single file for each image their corresponding captions as `Captions.csv` and for each 
image all the object classes as `Objects.csv`.
## Semantic Fidelity(SF) metric
```shell
python MetaData.py
```
The code calculates the Semantic Fidelity value for each captions and the final value are saved as `Meta-data.csv`.
### SF and its variants
To check the authenticity of the initial SF metric we comapre its performance with Human Semantic Fidelity and various 
different forms of SFs by comapring the pearson correlation and r2 values of these forms.
```shell
python SFs_plot.py
```
The code outputs various regression plots for the different variants of SFs with their corresponding confidence interval.
##  Final Caption
The Semantic Fidelity as calculated is used to output the top 3 captions for the given image as 
```shell
python final_caption.py --image ****.jpg
```
