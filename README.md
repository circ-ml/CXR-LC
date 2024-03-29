# CXR-LC: Deep learning using Chest Radiographs to Identify High-risk Smokers for Lung Cancer Screening: Development and Validation of a Prediction Model

![CXR-LC Grad-CAM](/images/GradCAM.png)

Original publication:

[Lu MT*, Raghu VK*, Mayrhofer T, Aerts HJWL, Hoffmann U. Deep learning using chest radiographs to identify high-risk smokers for lung cancer screening computed tomography: Development and validation of a prediction model. Annals of Internal Medicine 2020;173:704-713.](https://pubmed.ncbi.nlm.nih.gov/32866413/) *Equal contribution

[Editorial by Pinsky P. Artificial Intelligence and Data Mining to Assess Lung Cancer Risk: Challenges and Opportunities. Annals of Internal Medicine 2020;173(9):760-761.](https://pubmed.ncbi.nlm.nih.gov/32866415/)

Subsequent external validation:

[Lee JH, Lee D, Lu MT, Raghu VK, Park CM, Goo JM, Choi SH, Kim H. Deep learning to optimize candidate selection for lung cancer CT screening: Advancing the 2021 USPSTF Recommendations. Radiology 2022;305(1):209-218.](https://pubmed.ncbi.nlm.nih.gov/35699582/)

[Raghu VK, Walia AS, Zinzuwadia AN, Goiffon RJ, Shepard JO, Aerts HJWL, Lennes IT, Lu MT. Validation of a deep learning-based model to predict lung cancer risk using chest x-rays and electronic medical record data. JAMA Network Open 2022;5(12):e2248793.](https://pubmed.ncbi.nlm.nih.gov/36576736/)

## Overview
Lung cancer screening with chest CT reduces lung cancer death. Currently, CT screening is offered to those eligible by the Centers for Medicare and Medicaid services (CMS) screening criteria. This criteria misses many lung cancers and requires detailed smoking information (e.g. cigarette pack-years) that is poorly documented in the electronic medical record (EMR). Less than 5% of Americans eligible for lung cancer screening CT are screened, a dismal rate compared to breast (64%) and colorectal (63%) cancer screening. An automated way to flag high risk smokers could improve screening participation and prevent lung cancer death.

Chest x-rays (radiographs or CXRs) are among the most common diagnostic imaging tests in medicine. We hypothesized that a convolutional neural network (CNN) could extract information from these x-rays, along with other data commonly available in the electronic medical record, to identify individuals at high risk for lung cancer who would benefit from lung cancer screening CT.

Our CXR-LC CNN outputs a probability of 12-year incident lung cancer based on inputs of a chest radiograph image, age, sex, and smoking status (current vs. former smoker). CXR-LC was developed in 41,856 persons from the Prostate, Lung, Colorectal and Ovarian cancer screening trial (PLCO), a randomized controlled trial of chest x-ray to screen healthy persons for lung cancer. 

CXR-LC was tested (referred to as "validation" in the publication) in an independent cohort of 5,615 PLCO smokers and externally tested in 5,493 heavy smokers from the National Lung Screening Trial (NLST). CXR-LC performed better than the current clinical standard, the CMS eligibility criteria for lung cancer screening CT (AUC 0.755 vs 0.634, p < 0.001). When compared at equal-sized screening populations to the CMS criteria, CXR-LC missed 30.7% fewer lung cancers. 



**Inverted Kaplan-Meier for Incident Lung Cancer based on CXR-LC category**
![CXR-LC Kaplan-Meier](/images/KM_Curve_cropped.jpg)

This repo contains data intended to promote reproducible research. It is not for clinical care or commercial use. 

## Installation
This inference code was tested on Ubuntu 18.04.3 LTS, conda version 4.8.0, python 3.7.7, fastai 1.0.61, cuda 10.2, pytorch 1.5.1 and cadene pretrained models 0.7.4. A full list of dependencies is listed in `environment.yml`. 

Inference can be run on the GPU or CPU, and should work with ~4GB of GPU or CPU RAM. For GPU inference, a CUDA 10 capable GPU is required.

For the model weights to download, Github's large file service must be downloaded and installed: https://git-lfs.github.com/ 

This example is best run in a conda environment:

```bash
git lfs clone https://github.com/vineet1992/CXRLC/
cd location_of_repo
conda env create -n CXRLC -f environment.yml
conda activate CXRLC
python run_mixed.py dummy_datasets/test_images/ development/models/CXRLC dummy_datasets/Tabular_Data.csv output/output.csv --cont=age --cat=sex,smoke --target=is_lungcancer
```

Dummy data files are provided in `dummy_datasets/Tabular_Data.csv` and `dummy_datasets/test_images/;` dummy images are provided in the `test_images` folder. Weights for the CXR-Risk model are in `development/models/CXRLC.pth`. 

Age input into the model must be normalized by the following equation:

Model_Age = (age - 62.71093709214304) / 5.2805621041415804 .

Sex should be encoded as 0 = Male and 1 = Female .

Smoking Status should be encoded as 0 = current smoker and 1 = former smoker .

## Datasets
PLCO (NCT00047385) data used for model development and testing are available from the National Cancer Institute (NCI, https://biometry.nci.nih.gov/cdas/plco/). NLST (NCT01696968) testing data is available from the NCI (https://biometry.nci.nih.gov/cdas/nlst/) and the American College of Radiology Imaging Network (ACRIN, https://www.acrin.org/acrin-nlstbiorepository.aspx). Due to the terms of our data use agreement, we cannot distribute the original data. Please instead obtain the data directly from the NCI and ACRIN.

The `data` folder provides the image filenames, split of PLCO into independent training/tuning/testing datasets, and the CXR-Risk output probabilities: 
* `PLCO_Scaled_Probabilities.csv` contains the probabilities (Scaled_CXRLC_Risk) generated by CXR-LC in the PLCO testing dataset. Probabilities were put into ordinal categories (CXRLC_Group) as described in the manuscript. filenamepng refers to the original TIF filename converted to a PNG. "Dataset" is "Tr" for training, "Tu" for tuning, and "Te" for testing. 
* `NLST_Scaled_Probabilities.csv` contains the probabilities (Scaled_CXRLC_Risk) and ordinal probability categories (CXRLC_Group) for the NLST testing dataset. The format for filenamepng is the (original participant directory)_(original DCM filename).png


## Image processing
PLCO radiographs were provided as scanned TIF files by the NCI. TIFs were converted to PNGs with a minimum dimension of 512 pixels with ImageMagick v6.8.9-9. 

Many of the PLCO radiographs were rotated 90 or more degrees. To address this, we developed a CNN to identify rotated radiographs. First, we trained a CNN using the resnet34 architecture to identify synthetically rotated radiographs from the [CXR14 dataset](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf). We then fine tuned this CNN using 11,000 manually reviewed PLCO radiographs. The rotated radiographs identified by this CNN in `preprocessing/plco_rotation_github.csv` were then corrected using ImageMagick. 

```bash
cd path_for_PLCO_tifs
mogrify -path destination_for_PLCO_pngs -trim +repage -colorspace RGB -auto-level -depth 8 -resize 512x512^ -format png "*.tif"
cd path_for_PLCO_pngs
while IFS=, read -ra cols; do mogrify -rotate 90 "${cols[0]}"; done < /path_to_repo/preprocessing/plco_rotation_github.csv
```

NLST radiographs were provided as DCM files by ACRIN. We chose to first convert them to TIF using DCMTK v3.6.1, then to PNGs with a minimum dimension of 512 pixels through ImageMagick to maintain consistency with the PLCO radiographs:

```bash
cd path_to_NLST_dcm
for x in *.dcm; do dcmj2pnm -O +ot +G $x "${x%.dcm}".tif; done;
mogrify -path destination_for_NLST_pngs -trim +repage -colorspace RGB -auto-level -depth 8 -resize 512x512^ -format png "*.tif"
```


The orientation of several NLST chest radiographs was manually corrected:

```
cd destination_for_NLST_pngs
mogrify -rotate "90" -flop 204025_CR_2000-01-01_135015_CHEST_CHEST_n1__00000_1.3.51.5146.1829.20030903.1123713.1.png
mogrify -rotate "-90" 208201_CR_2000-01-01_163352_CHEST_CHEST_n1__00000_2.16.840.1.113786.1.306662666.44.51.9597.png
mogrify -flip -flop 208704_CR_2000-01-01_133331_CHEST_CHEST_n1__00000_1.3.51.5146.1829.20030718.1122210.1.png
mogrify -rotate "-90" 215085_CR_2000-01-01_112945_CHEST_CHEST_n1__00000_1.3.51.5146.1829.20030605.1101942.1.png
```
As a final step, we used the Python Imaging Library (PIL) to resize each image to 224 x 224 using a bilinear interpolation.

```
python preprocessing/ImageResizer.py <path_to_512_images> <path_to_output_directory_224_224_images>
```

## Acknowledgements
I thank the NCI and ACRIN for access to trial data, as well as the PLCO and NLST participants for their contribution to research. I would also like to thank the fastai and Pytorch communities. A GPU used for this research was donated as an unrestricted gift through the Nvidia Corporation Academic Program. The statements contained herein are mine alone and do not represent or imply concurrence or endorsements by the above individuals or organizations.


