% function for bsl to find the best start pair
function final=findstart(num,z)
[s1,s2,bandnum]=size(z);
bstart=zeros(1,num);
final=zeros(1,num);
%allbsn=zeros(158,num);
for k=1:num
bstart=zeros(1,num);
bstart(1)=k;
start=k;
for i=1:num
sz=reshape(z,s1*s2,bandnum);
X=sz(:,start);
sz(:,start)=[];

    e=zeros(1,bandnum-1);
    for j=1:bandnum-1
    y=sz(:,j);
    b=inv(X'*X)*X'*y;
    y_hat=X*b;
    r=y-y_hat;
    e(1,j)=norm(r);
    end
    [v,index]=max(e);
    start=index+(index>start);
    
    
    
    if (sum(bstart==start))
        final(k)=start;
        break;
    end
    bstart(i+1)=start;
    
end
end

