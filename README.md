# V2T-Similarity-IMAVIS-2023
## Description
This is the implementation of our paper entitled [**"Improving distinctiveness in video captioning with text-video similarity"**](https://www.sciencedirect.com/science/article/pii/S0262885623001026).

Our approach enhances the distinctiveness of video captioning by integrating video retrieval into the training process. We calculate similarity scores between the generated text and videos, incorporating them into the training loss. Additionally, we use reference scores, representing similarity between ground truth sentences and videos, to scale the training loss. This guides the model to generate sentences that closely match the desired level of distinctiveness indicated by the reference scores.

It is demonstrated in the experiments of MSVD and MSR-VTT that our method improved video captioning quantitatively and qualitatively.

The illustration of our proposed method is shown below:
![alt text](/assets/architecture.png)
## Prepare the Environment 
Install and create conda environment with the provided `environment.yml` file.
This conda environment was tested with the NVIDIA RTX 3090.

The details of each dependency can be found in the environment.yml file.
```
conda env create -f environment.yml
conda activate rl
pip install git+https://github.com/Maluuba/nlg-eval.git@master
pip install pycocoevalcap

Install torch following this page: https://pytorch.org/get-started/locally 
pip install opencv-python
pip install seaborn
pip install boto3
pip install ftfy
pip install h5py

```


## Prepare the Dataset

### Dataset Folder Structure
```bash
├── dataset
│   ├── MSVD
│   │   ├── raw # put the 1970 raw videos in here
│   │   ├── captions 
│   │   ├── raw-captions_mapped.pkl # mapping between video id with captions
│   │   ├── train_list_mapping.txt
│   │   ├── val_list_mapping.txt
│   │   ├── test_list_mapping.txt
│   ├── MSRVTT
│   │   ├── raw # put the 10000 raw videos in here
│   │   ├── msrvtt.csv # list of video id in msrvtt dataset
│   │   ├── MSRVTT_data.json # metadata of msrvtt dataset, which includes video url, video id, and caption
```
### MSR-VTT
Raw videos can be downloaded from this [link](https://github.com/VisionLearningGroup/caption-guided-saliency/issues/6).
### MSVD

Raw videos can be downloaded from this [link](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/).

## Download Extracted Features
1. Download the extracted features from [link](https://drive.google.com/drive/folders/1RxJoUlBffWpdsOJJI8SfXIIRJMxZvUCb?usp=sharing).
2. Put the extracted features into ./features folder.



## Training
In our paper, we use [CLIP4Caption](https://dl.acm.org/doi/10.1145/3474085.3479207) and [CLIP4Clip](https://arxiv.org/abs/2104.08860) as our video captioning and video retrieval, respectively. 

### Download Video Captioning
1. Clone our implementation of CLIP4Caption to the root folder.
```
git clone https://github.com/Sejong-VLI/V2T-CLIP4Caption-Reproduction.git
```
2. Rename the folder as you want.
3. Modify the import library in ```<VIDEOCAPTIONINGFOLDER>/modules/modeling.py``` as follows:
```
from <VIDEOCAPTIONINGFOLDER>.modules.until_module import PreTrainedModel, LayerNorm, CrossEn
from <VIDEOCAPTIONINGFOLDER>.modules.module_bert import BertModel, BertConfig
from <VIDEOCAPTIONINGFOLDER>.modules.module_visual import VisualModel, VisualConfig, VisualOnlyMLMHead
from <VIDEOCAPTIONINGFOLDER>.modules.module_decoder import DecoderModel, DecoderConfig
```

4. Modify the import library in ```<VIDEOCAPTIONINGFOLDER>/modules/until_module.py``` as follows:
```
from <VIDEOCAPTIONINGFOLDER>.modules.until_config import PretrainedConfig
```

5. Change ```<VIDEOCAPTIONINGFOLDER>/dataloaders``` with the provided ```dataloaders``` folder.

**Note** Please replace the ```<VIDEOCAPTIONINGFOLDER>``` with the folder name you have chosen. For example, if your folder name is CLIP4Caption then the import library in **policy_gradient.py** will be ```from CLIP4Caption.modules.tokenization import BertTokenizer```.


### Download Video Retrieval
1. Clone the implementation of CLIP4Clip to the root folder.
```
git clone https://github.com/ArrowLuo/CLIP4Clip.git
```
2. Rename the folder as you want.
3. Change ```<VIDEORETRIEVALFOLDER>/modules/modeling.py``` with the provided ```modeling.py```.
4. Change ```<VIDEORETRIEVALFOLDER>/modules/tokenization_clip.py``` with the provided ```tokenization_clip.py```.

5. Modify the import library in ```<VIDEORETRIEVALFOLDER>/modules/until_module.py``` as follows:
```
from <VIDEORETRIEVALFOLDER>.modules.until_config import PretrainedConfig
```
**Note** Please replace the ```<VIDEORETRIEVALFOLDER>``` with the folder name you have chosen. For example, if your folder name is CLIP4Clip then the import library in **<VIDEORETRIEVALFOLDER>/modules/until_module.py** will be ```from CLIP4Clip.modules.until_config import PretrainedConfig```.

### Final Folder Structure
The folder structure after downloading the video captioning and video retrieval should look as follows:
```bash
├── <VIDEOCAPTIONINGFOLDER>
├── <VIDEORETRIEVALFOLDER>  
├── dataset
├── features
├── pretrained
├── environment.yml
├── policy_gradient.py
├── train.py
├── converter.py
├── retrieval_utils.py
```
### Pretraining Video Retrieval
Download pretrained model from [link](https://drive.google.com/drive/folders/141yGBfxfLwbgzXXKZ74LegeZBy2oooi7?usp=sharing) and put into ./pretrained folder.

### Training the Video Captioning
1. Initialize our caption generator.
```
mkdir -p ./<VIDEOCAPTIONINGFOLDER>/weight
wget -P ./<VIDEOCAPTIONINGFOLDER>/weight https://github.com/microsoft/UniVL/releases/download/v0/univl.pretrained.bin
```
**Note** Please replace the ```<VIDEOCAPTIONINGFOLDER>``` with the folder name you have chosen. For example, if your video captioning folder name is CLIP4Caption then the command will be ```mkdir -p ./CLIP4Caption/weight```.

2. In each train script (.sh), change following parameters based on the specs of your machine and the data location:
    - **N_GPU** = [Total GPU to use]
    - **N_THREAD** = [Total thread to use]
    - **DATA_PATH** = [JSON file location]
    - **CKPT_ROOT** = [Your desired folder for saving the models and results]
    - **INIT_MODEL_PATH** = [UniVL pretrained model location]
    - **FEATURES_PATH** = [Generated video features path]
    - **MODEL_FILE_RET** = [Pretrain video retrieval checkpoint]
    - **MODEL_FILE** = [Saved video captioning model for evaluation]
3. Execute the following scripts to start the training process.
4. Run following script:
```
python3 converter.py --replace_variable='<VIDEOCAPTIONINGFOLDER>' --target_variable='<VIDEOCAPTIONINGFOLDER>'
python3 converter.py --replace_variable='<VIDEORETRIEVALFOLDER>' --target_variable='<VIDEORETRIEVALFOLDER>'
```

For example if your **video captioning folder** name is CLIP4Caption then the script will become:
```
 python3 converter.py --replace_variable='CLIP4Caption' --target_variable='<VIDEOCAPTIONINGFOLDER>'
```

##### Training Using MSVD
```
cd scripts/
./msvd_train.sh 
```
##### Training Using MSRVTT
```
cd scripts/
./msrvtt_train.sh  
```
## Evaluation
After the training is done, an evaluation process on the test set will be automatically executed using the best checkpoint among all epochs. However, if you want to evaluate a checkpoint at a specific epoch, you can use the provided training shell script by modifying the value of `INIT_MODEL_PATH` to the location of the desired checkpoint, and replacing the `--do_train` to `--do_eval`.

## Our Results

The comparison with the existing methods and also the ablation study of our method can be found in our paper.

### MSVD

| Method  | CLIP Model | BLEU@4 | METEOR | ROUGE-L | CIDEr | R@1 |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |
| Ours | ViT-B/16 | 64.77 | 42.05 | 78.77 | 124.47 | 30.8

#### MSR-VTT

| Method  | CLIP Model | BLEU@4 | METEOR | ROUGE-L | CIDEr | R@1 |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |
| Ours | ViT-B/16 | 48.78 | 31.28 | 65.01 | 60.51 | 17.0

## Acknowledgements
Our code is developed based on https://github.com/microsoft/UniVL, which is also developed based on https://github.com/huggingface/transformers/tree/v0.4.0 and https://github.com/antoine77340/howto100m .

## Citation
Please cite our paper in your publications if it helps your research as follows:
```
@article{VELDA2023104728,
title = {Improving distinctiveness in video captioning with text-video similarity},
journal = {Image and Vision Computing},
pages = {104728},
year = {2023},
issn = {0262-8856},
doi = {https://doi.org/10.1016/j.imavis.2023.104728},
url = {https://www.sciencedirect.com/science/article/pii/S0262885623001026},
author = {Vania Velda and Steve Andreas Immanuel and Willy Fitra Hendria and Cheol Jeong}
}
```


