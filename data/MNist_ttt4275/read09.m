% This program will take some time ( approx 15 minutes?) but should be used only once. Use the Matlab file data_all after this
fid =  fopen('train_images.bin','r');
magic_num = fread(fid,1,'int32','ieee-be');
num_train=fread(fid,1,'int32','ieee-be');
row_size=fread(fid,1,'int32','ieee-be');
col_size=fread(fid,1,'int32','ieee-be');

vec_size = row_size*col_size;
 trainv=zeros(num_train,vec_size);

for k = 1:num_train
	for n = 1:vec_size
        	 trainv(k,n) =fread(fid,1,'uchar','ieee-be');
	end
end

fclose(fid);

fid =  fopen('test_images.bin','r');
magic_num = fread(fid,1,'int32','ieee-be');
num_test=fread(fid,1,'int32','ieee-be');
row_size=fread(fid,1,'int32','ieee-be');
col_size=fread(fid,1,'int32','ieee-be');

vec_size = row_size*col_size;
 testv=zeros(num_test,vec_size);

for k = 1:num_test
	for n = 1:vec_size
        	testv(k,n) =fread(fid,1,'uchar','ieee-be');
	end
end

fclose(fid);

disp('labels')
fid =  fopen('train_labels.bin','r');
magic_num = fread(fid,1,'int32','ieee-be');
num_train=fread(fid,1,'int32','ieee-be');
 
trainlab=zeros(num_train,1);

for k = 1:num_train
	trainlab(k) =fread(fid,1,'uchar','ieee-be');
end

fclose(fid);

fid =  fopen('test_labels.bin','r');
magic_num = fread(fid,1,'int32','ieee-be');
num_test=fread(fid,1,'int32','ieee-be');

testlab = zeros(num_test,1);

for k = 1:num_test
	testlab(k) =fread(fid,1,'uchar','ieee-be');
end

fclose(fid);

save data_all num_train num_test row_size col_size vec_size trainv  trainlab testv  testlab