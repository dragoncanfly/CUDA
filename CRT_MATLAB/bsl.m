
% band selection use MLR
function bsn=bsl(z,ind,num)
% read the size from image
[s1,s2,bandnum]=size(z);
% reshape the data to 2d matrix
z=reshape(z,s1*s2,bandnum);
% set up initial band and the remain for select
remain=1:bandnum;
X=z(:,ind);
remain(ind)=[];
% do forward search in the remain set
% f = waitbar(0,'begin band seletion operation...');
nbands=num-length(ind);
for i=1:nbands
    % use normal equation to do the multiple linear regression
    e=0;
    for j=1:length(remain)
      ne=bandlp(z,ind,remain(j));
      if ne>e
        e=ne;
        index=j;
      end
    end
    % find the largest error to denote the most dissimilarity
    
    ind=[ind,remain(index)];
    remain(index)=[];
   
   %  waitbar(i/nbands,f,strcat('Progress: ',num2str(i),'/',num2str(nbands)));
end
bsn=ind;
% close(f);