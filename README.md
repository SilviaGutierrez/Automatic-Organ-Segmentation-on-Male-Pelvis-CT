# Automatic-Organ-Segmentation-on-Male-Pelvis-CT
MATLAB based application developed to automate the segmentation
process of the prostate, bladder, and rectum in male pelvic CT images
# Abstract
Prostate cancer is a prevalent and life-threatening disease, ranking as the second most common cancer and the fifth leading cause of cancer-related deaths
in men. Precise segmentation of the prostate and surrounding organs at risk (OARs) on computed tomography (CT) images is crucial for effective radiotherapy treatment planning. However, manual segmentation is a time-consuming
task that is prone to interobserver and intraobserver variabilities. An innovative MATLAB-based application is developed to automate the
segmentation process of the prostate, bladder, and rectum in male pelvic CT images. The automatic segmentation app uses a pre-trained ResNet-18 convolutional
neural network (CNN). The ResNet-18 CNN, combined with an encoder-decoder structure based on the DeepLabv3+ model, achieves accurate and efficient segmentation without the need for preprocessing techniques or extensive manual refinement.
The proposed method was evaluated using a dataset of 100 CT scans, and promising results were obtained in terms of the Dice similarity coefficient
(DSC) for prostate segmentation. The average DSC was 87.8% for segmenting the prostate using a single slice per patient and 86.2% for segmenting the
prostate using three slices per patient. Furthermore, the proposed method was extended to segment the prostate, bladder, and rectum, achieving mean DSC values ranging from 71.4% to 80.1% for prostate segmentation, 73.4% to 86.5%
for bladder segmentation, and 80.5% to 88.2% for rectum segmentation. The automatic segmentation app based on the ResNet-18 CNN offers significant advancements in radiotherapy treatment planning for prostate cancer. It reduces the burden on physicians, improves efficiency and accuracy, and
minimizes interobserver and intraobserver variabilities. By providing precise segmentation of the prostate and OARs, the method enables better dose calculation, treatment assessment, and reduction of radiation-related side effects.
# Male Pelvis CT database
A total of 100 radiotherapy planning CT scans were collected from 100 patients at the Juan Ramon Jimenez Hospital, Huelva, Spain, between 2018 and 2020, following
100 approval from the institutional ethics committee. The radiation oncologists contoured the prostate, bladder, and rectum, which served
as the ground truth for the automatic segmentation task.

![image](https://github.com/SilviaGutierrez/Automatic-Organ-Segmentation-on-Male-Pelvis-CT/assets/108027382/03a0ce0e-5c8c-4722-8873-6e26d87a9eb2)






