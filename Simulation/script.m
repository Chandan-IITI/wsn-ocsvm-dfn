clear all;close all;clc;
load train_data;
trainData=trainData(:,2:3);
trainData=bsxfun(@rdivide,...
    trainData-repmat(min(trainData),size(trainData,1),1),...
    max(trainData)-min(trainData));

trainData=consolidator(trainData,[],@mean,1e-2);
trainLabel=ones(size(trainData,1),1);

s=ocsvm_dfn(trainData);
SVMModel=fitcsvm(trainData,trainLabel,'KernelScale',s,'OutlierFraction',0);

%
testData=rand(1e5,2);
[~,score]=predict(SVMModel,testData);
figure;
plot(trainData(:,1),trainData(:,2),'g.','MarkerSize',3);
hold on;
plot(SVMModel.SupportVectors(:,1),SVMModel.SupportVectors(:,2),'ko','MarkerSize',3);
hold on;
plot(testData(score<=0,1),testData(score<=0,2),'r.','MarkerSize',3);





