# COCOText
mkdir -p ocr-dataset && cd ocr-dataset

(
mkdir -p COCO && cd COCO
wget https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip
wget http://images.cocodataset.org/zips/train2014.zip
)

# ArT
(
mkdir -p ArT && cd ArT
#* manualy download from https://rrc.cvc.uab.es/?ch=14&com=downloads
)

# TextOCR
(
mkdir -p TextOCR && cd TextOCR
wegt https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json
wget https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
)

# ICDAR13:
( # Please check https://mmocr.readthedocs.io/en/latest/datasets/det.html?highlight=icdar#icdar-2013-focused-scene-text for details
mkdir icdar2013 && cd icdar2013
mkdir imgs && mkdir annotations

# Download ICDAR 2013
wget https://rrc.cvc.uab.es/downloads/Challenge2_Training_Task12_Images.zip --no-check-certificate
wget https://rrc.cvc.uab.es/downloads/Challenge2_Test_Task12_Images.zip --no-check-certificate
wget https://rrc.cvc.uab.es/downloads/Challenge2_Training_Task1_GT.zip --no-check-certificate
wget https://rrc.cvc.uab.es/downloads/Challenge2_Test_Task1_GT.zip --no-check-certificate

# For images
unzip -q Challenge2_Training_Task12_Images.zip -d imgs/training
unzip -q Challenge2_Test_Task12_Images.zip -d imgs/test
# For annotations
unzip -q Challenge2_Training_Task1_GT.zip -d annotations/training
unzip -q Challenge2_Test_Task1_GT.zip -d annotations/test

rm Challenge2_Training_Task12_Images.zip && rm Challenge2_Test_Task12_Images.zip && rm Challenge2_Training_Task1_GT.zip && rm Challenge2_Test_Task1_GT.zip
)