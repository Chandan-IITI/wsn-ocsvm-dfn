clear all;close all;clc;
addpath(genpath(pwd));

% Data
load train_data;
trainData=trainData(:,2:3);
normParam.min=min(trainData);
normParam.max=max(trainData);

trainData=bsxfun(@rdivide,...
    trainData-repmat(min(trainData),size(trainData,1),1),...
    max(trainData)-min(trainData));

trainData=consolidator(trainData,[],@mean,1e-2);
trainLabel=ones(size(trainData,1),1);

% DFN
s=ocsvm_dfn(trainData);

% OCSVM
mat2svm([trainLabel trainData]);
[trainLabel,trainData]=libsvmread('mySVMdata.txt');
options=sprintf('-s 2 -n 0.0001 -g %f',1/2/s^2);
ocsvmModel=svmtrain(trainLabel,trainData,options);

save ocsvm_model ocsvmModel normParam;

%%
clear all;close all;clc;
load ocsvm_model;
testData=repmat(normParam.min-10,1e6,1)+...
    bsxfun(@times,rand(1e6,2),(normParam.max-normParam.min+20));
[predictLabel,decValues]=ocsvm_classify(ocsvmModel,normParam,testData);

figure(1);clf;
plot(ocsvmModel.SVs(:,1)*(normParam.max(1)-normParam.min(1))+normParam.min(1),...
     ocsvmModel.SVs(:,2)*(normParam.max(2)-normParam.min(2))+normParam.min(2),'ro','MarkerSize',5);
hold on;
load train_data;
trainData=trainData(:,2:3);
plot(trainData(:,1),trainData(:,2),'g.','MarkerSize',3);
hold on;
plot(testData(abs(decValues)<=1e-4,1),testData(abs(decValues)<=1e-4,2),'k.','MarkerSize',3);
hold on;
plot(testData(abs(decValues+1e-2)<=1e-4,1),testData(abs(decValues+1e-2)<=1e-4,2),'k.','MarkerSize',3);
hold on;
plot(testData(abs(decValues+2e-2)<=1e-4,1),testData(abs(decValues+2e-2)<=1e-4,2),'k.','MarkerSize',3);
hold on;
plot(testData(abs(decValues+3e-2)<=1e-4,1),testData(abs(decValues+3e-2)<=1e-4,2),'k.','MarkerSize',3);

supportVector=ocsvmModel.SVs.*(normParam.max-normParam.min)+normParam.min;
boundaryData0=testData(abs(decValues)<=1e-4,:);
boundaryData1=testData(abs(decValues+1e-2)<=1e-4,:);
boundaryData2=testData(abs(decValues+2e-2)<=1e-4,:);
boundaryData3=testData(abs(decValues+3e-2)<=1e-4,:);
save data_2d supportVector trainData boundaryData0 boundaryData1 boundaryData2 boundaryData3;

%%
clear all;close all;clc;
load ocsvm_model;
ocsvmModel.rho=ocsvmModel.rho-2e-2;

clear ibrlData;
load ibrl_data;
index=date*24*60*60+time;
ibrlData=[month index moteid temperature humidity];
ibrlData(ibrlData(:,1)~=3,:)=[];  

% 
clear predictLabel moteData;
for i=[1 2 33 35 37]
    moteData{i}=ibrlData(ibrlData(:,3)==i,[2 4 5]);
    predictLabel{i}=ocsvm_classify(ocsvmModel,normParam,moteData{i}(:,[2 3]));
    figure(i);clf;
    plot(moteData{i}(:,[2 3]),'b-');
    hold on;
    plot(find(predictLabel{i}==-1),moteData{i}(predictLabel{i}==-1,[2 3]),...
        'ro','linewidth',2);
end

moteData=moteData{i}(:,[2 3]);
anomalyIndex=find(predictLabel{i}==-1);
anomalyData=moteData(anomalyIndex,:);
save timeData37 moteData anomalyIndex anomalyData;




