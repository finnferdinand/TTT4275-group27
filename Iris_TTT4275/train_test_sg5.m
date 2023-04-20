x1all = load('class_1','-ascii');
x2all = load('class_2','-ascii');
x3all = load('class_3','-ascii');

% x1= [x1all(:,4) x1all(:,1) x1all(:,2)];
% x2= [x2all(:,4) x2all(:,1) x2all(:,2)];
% x3= [x3all(:,4) x3all(:,1) x3all(:,2)];

%x1= [x1all(:,3) x1all(:,4)];
%x2= [x2all(:,3) x2all(:,4)];
%x3= [x3all(:,3) x3all(:,4)];

x1= [x1all(:,4)];
x2= [x2all(:,4)];
x3= [x3all(:,4)];


[Ntot,dimx] = size(x1);

for k = 1:5
	N1d = (k-1)*10 +1; N2d = rem((k+2)*10-1, 50)+1; N1t = rem(N2d+1,50); N2t = rem(N2d+19,50)+1;
	if(N2d < N1d)
		x1d = [x1(N1d:Ntot,:); x1(1:N2d,:)]; x2d = [x2(N1d:Ntot,:); x2(1:N2d,:)]; x3d = [x3(N1d:Ntot,:); x3(1:N2d,:)];
	else
		x1d = x1(N1d:N2d,:); x2d = x2(N1d:N2d,:); x3d = x3(N1d:N2d,:);
	end
	if(N2t < N1t)
		x1t = [x1(N1t:Ntot,:); x1(1:N2t,:)]; x2t = [x2(N1t:Ntot,:); x2(1:N2t,:)]; x3t = [x3(N1t:Ntot,:); x3(1:N2t,:)];
	else
		 x1t = x1(N1t:N2t,:); x2t = x2(N1t:N2t,:); x3t = x3(N1t:N2t,:);
	end
	
	Ndtot = 30; Nttot = Ntot - Ndtot;

 

	x1m = mean(x1d); x1s = std(x1d);
	x2m = mean(x2d); x2s = std(x2d);
	x3m = mean(x3d); x3s = std(x3d);

	indd =zeros(Ndtot,3);

	y1d =zeros(Ndtot,3);
	y1d(:,1) = mvnpdf(x1d,x1m, x1s);
	y1d(:,2) = mvnpdf(x1d,x2m, x2s);
	y1d(:,3) = mvnpdf(x1d,x3m, x3s);
	[val1d,indd(:,1)] = max(y1d');

	y2d =zeros(Ndtot,3);
	y2d(:,1) = mvnpdf(x2d,x1m, x1s);
	y2d(:,2) = mvnpdf(x2d,x2m, x2s);
	y2d(:,3) = mvnpdf(x2d,x3m, x3s);
	[val2d,indd(:,2)] = max(y2d');

	y3d =zeros(Ndtot,3);
	y3d(:,1) = mvnpdf(x3d,x1m, x1s);
	y3d(:,2) = mvnpdf(x3d,x2m, x2s);
	y3d(:,3) = mvnpdf(x3d,x3m, x3s);
	[val3d,indd(:,3)] = max(y3d');

	confd = zeros(3,3);

	for i = 1:3 % correct class
		for j = 1:3 % chosen class
			 confd(i,j)= length(find(indd(:,i) == j));
		end
	end

	indt =zeros(Nttot,3);

	y1t =zeros(Nttot,3);
	y1t(:,1) = mvnpdf(x1t,x1m, x1s);
	y1t(:,2) = mvnpdf(x1t,x2m, x2s);
	y1t(:,3) = mvnpdf(x1t,x3m, x3s);
	[val1t,indt(:,1)] = max(y1t');

	y2t =zeros(Nttot,3);
	y2t(:,1) = mvnpdf(x2t,x1m, x1s);
	y2t(:,2) = mvnpdf(x2t,x2m, x2s);
	y2t(:,3) = mvnpdf(x2t,x3m, x3s);
	[val2t,indt(:,2)] = max(y2t');

	y3t =zeros(Nttot,3);
	y3t(:,1) = mvnpdf(x3t,x1m, x1s);
	y3t(:,2) = mvnpdf(x3t,x2m, x2s);
	y3t(:,3) = mvnpdf(x3t,x3m, x3s);
	[val3t,indt(:,3)] = max(y3t');

	conft = zeros(3,3);

	for i = 1:3 % correct class
		for j = 1:3 % chosen class
			 conft(i,j)= length(find(indt(:,i) == j));
		end
	end

	disp([confd conft])

end