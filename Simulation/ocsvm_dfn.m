function s=ocsvm_dfn(trainData)

    n=size(trainData,1);
    for i=1:n
        X1=trainData;
        X1(i,:)=[];
        [~,D] = knnsearch(X1,trainData(i,:));
        dmin(i)=D;
        dmaxi=0;
        for j=1:n
            dmaxij=norm(trainData(i,:)-trainData(j,:));
            if dmaxij>dmaxi
               dmaxi=dmaxij;
            end
        end
        dmax(i)=dmaxi;
    end

    s0=rand(1);
    options=optimoptions('fminunc','SpecifyObjectiveGradient',true,...
        'Display','iter','Algorithm','trust-region',...
        'OptimalityTolerance',1e-8,'PlotFcn',@optimplotfval);
    [s,fval,exitflag,output]=fminunc(@(s)obj_fcn(dmin,dmax,s),s0,options);
    s=sqrt(s/2);

end

function [f,g]=obj_fcn(dmin,dmax,s)

    n=numel(dmin);
    f=0;g=0;
    for i=1:n
        f=f-2/n*exp(-dmin(i)/s)+2/n*exp(-dmax(i)/s);
        g=g-2/n*exp(-dmin(i)/s)*dmin(i)/s^2+2/n*exp(-dmax(i)/s)*dmax(i)/s^2;
    end

end


