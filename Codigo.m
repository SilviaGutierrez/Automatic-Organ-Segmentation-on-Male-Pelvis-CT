%%%% Code for binary segmentation model; prostate and background
%%% Preparing the dataset. The dataset is composed of images and labeled pixels maps
path='...'; % dataset path. 
%% load images data 
ImgDir=fullfile(path,'ImageSubFolderName'); 
ImgDs=imageDatastore(ImgDir,'FileExtensions','.dcm','ReadFcn',...
@(x)dicomread(x)); 
%% Read image
I=readimage(ImgDs,n); %% read the nth image
%% load labels data
LabDir=fullfile(DataSetDir,'Labels'); 
classNames=["Background","Prostate"]; %%%% classNames=["Background","Bladder","Rectum","Prostate"] for multiclass segmentation model
pixelLabelID = [0 1]; %%%% pixelLabelID = [0 1 2 3] for multiclass segmentation model 
PxDs=pixelLabelDatastore(LabDir,classNames,pixelLabelID,'FileExtensions','.dc
m','ReadFcn',@(x) uint8(dicomread(x))); 
%% 10-fold cross validation 
c=cvpartition(100,'KFold',10); 
for k=1:10 
ValIdx(k,:)=test(c,k); 
TralIdx(k,:)=training(c,k); 
end 
%% Prepararing training and validation data
TrainImages = ImgDs.Files(TrainIdx(k,:)); % for each k-interaction 90 images are extracted for training
ValImages = ImgDs.Files(ValIdx(k,:)); % for each interaction 10 images are extracted for validation
ImgDsTrain =imageDatastore(TrainImages,'FileExtensions','.dcm','ReadFcn',@(x) 
dicomread(x)); 
ImgDsVal = imageDatastore(ValImages,'FileExtensions','.dcm','ReadFcn',@(x) 
dicomread(x)); 
TrainLabels = PxDs.Files(TrainIdx(k,:)); % for each k-interaction 90 labels pixels map are extracted for training
ValLabels = PxDs.Files(ValIdx(k,:)); % for each k-interaction 10 labels pixels map are extracted for training
PxDsTrain = pixelLabelDatastore(TrainLabels, classNames, 
pixelLabelID,'FileExtensions','.dcm','ReadFcn',@(x) uint8(dicomread(x))); 
PxDsVal = pixelLabelDatastore(ValLabels, classNames, 
pixelLabelID,'FileExtensions','.dcm','ReadFcn',@(x) uint8(dicomread(x)));
                         
%%% Network setup and training options
imageSize=[512,512]; 
numClasses=2; %%%% numClasses=4 for multiclass segmentation model
lgraph = deeplabv3plusLayers(imageSize,numClasses,'resnet18'); 
%% training options 
ValDs = pixelLabelImageDatastore(ImgDsVal,PxDsVal,'OutputSize',imageSize,...
'ColorPreprocessing','gray2rgb'); 
options = trainingOptions('adam','MaxEpochs',10,'InitialLearnRate',1e-4,'LearnRateSchedule','piecewise', ...  
'LearnRateDropPeriod',20,'LearnRateDropFactor',0.95,'ValidationData',ValDs,'ValidationFrequency',90, ... 
 'Plots','training-progress','Verbose',false,'ValidationPatience', Inf,'Shuffle','every-epoch', ...
 'CheckpointPath', tempdir,'MiniBatchSize',1); 
%% Training 
TrainDs = pixelLabelImageDatastore(ImgDsTrain,PxDsTrain,'OutputSize',imageSize,...
'ColorPreprocessing','gray2rgb');  
[net, info] = trainNetwork(TrainDs,lgraph,options); 
%%% Evaluate segmentation results
PxDsResults = semanticseg(ValDs,net,"WriteLocation",tempdir);  
%% Metrics
metrics = evaluateSemanticSegmentation(PxDsResults,PxDsVal,'Verbose',false); 
metrics.DataSetMetrics; 
metrics.ClassMetrics; 
metrics.ConfusionMatrix; % Devuelve la matriz de confusion
normConfMatData = metrics.NormalizedConfusionMatrix.Variables; 
figure 
h = heatmap(classNames,classNames,100*normConfMatData); 
h.XLabel = 'Predicted Class'; 
h.YLabel = 'True Class'; 
h.Title = 'Resnet18 Normalized Confusion Matrix (%)'; 
%% Dice Similarity Coefficient 
for j=1:length(ImgDsVal.Files) 
LabelVer=readimage(PxDsVal,j); 
LabelPre = readimage(PxDsResults,j); 
diceResult(j,:) = dice(LabelVer,LabelPre); 
                         


