%% Time-domain validation
%
clear ibrlData;
load ibrl_data;
index=date*24*60*60+time;
ibrlData=[month index moteid temperature humidity];
ibrlData(ibrlData(:,1)~=3,:)=[];  

% 
clear predictLabel moteData;
for i=37
    moteData{i}=ibrlData(ibrlData(:,3)==i,[2 4 5]);
    predictLabel{i}=svdd_classify(ocSVM,moteData{i}(:,[2 3]));
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