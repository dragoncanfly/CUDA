function  DataLBP2 = LBP_Feature_Extraction(Data,height,width)

DataImage(2:height+1,2:width+1)=Data;
DataImage(1,2:width+1)=Data(1,:);
DataImage(height+2,2:width+1)=Data(height,:);
DataImage(2:height+1,1)=Data(:,1);
DataImage(2:height+1,width+2)=Data(:,width);
DataImage(1,1)=Data(1,1);
DataImage(1,width+2)=Data(1,width);
DataImage(height+2,1)=Data(height,1);
DataImage(height+2,width+2)=Data(height,width);
[m,n]=size(DataImage);
DataLBP=zeros(m,n);
    for i=2:m-1 
        for j=2:n-1
            center=DataImage(i,j);
            left_top=DataImage(i-1,j-1);
            val=double(center<left_top);
            top=DataImage(i-1,j);
            val=val+bitshift(double(center<top),1);
            right_top=DataImage(i-1,j+1);
            val=val+bitshift(double(center<right_top),2);
            right=DataImage(i,j+1);
            val=val+bitshift(double(center<right),3);
            right_bottom=DataImage(i+1,j+1);
            val=val+bitshift(double(center<right_bottom),4);
            bottom=DataImage(i+1,j);
            val=val+bitshift(double(center<bottom),5);
            left_bottom=DataImage(i+1,j-1);
            val=val+bitshift(double(center<left_bottom),6);
            left=DataImage(i,j-1);
            val=val+bitshift(double(center<left),7);
            DataLBP(i,j)=val;
        end
    end
    DataLBP2=DataLBP(2:m-1,2:n-1);
end

