%%%测试BP算法
%%% 比较不同学习率的效果
%%%%% 比较自适应学习率的效果
close('all');
clear;

%%%%%隐节点的设定
% k=2;%%%两个隐节点
k=4;%%%四个隐节点
%%%%%%%%%%%%%%%%%

%%%%%%%学习率的设定
% lr=.01; %learning rate %%%学习率
lr=.0001; %learning rate %%%学习率
%%%%%%%%%%%%

%%%%%%%自适应学习率的设定
% code=1; %Code for the chosen training algorithm%%%% 1是BP,2是惯量型的，3是自适应学习率
% par_vec=[lr 0 0 0 0];
code=3; %Code for the chosen training algorithm%%%% 1是BP,2是惯量型的，3是自适应学习率
par_vec=[lr 0 1.05 0.7 1.04]; %Parameter vector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%生成数据集部分
randn('seed',0);
% 1. Generate data set X1 
l=2; %Dimensionality
m1=[-5 5; 5 -5; 10 0]'; % centroids%%%%第一类中心，三个
m2=[-5 -5; 0 0; 5 5; 15 -5]';%第二类中心，四个
[l,c1]=size(m1); %no of gaussians per class%%%每类高斯分布的个数
[l,c2]=size(m2);%%每类高斯分布的个数

P1=ones(1,c1)/c1; % weights of the mixture model
P2=ones(1,c2)/c2;
s=1; % variance

% Generate the training data from the first class
N1=60; %Number of first class data points%%%%第一类的容量
for i=1:c1
    S1(:,:,i)=s*eye(l);
end
sed=0; %Random generator seed
[class1_X,class1_y]=mixt_model(m1,S1,P1,N1,sed);%%%%生成第一类训练样本

% Generate the training data from the second class%%生成第二类训练样本
N2=80; %Number of second class data points
for i=1:c2
    S2(:,:,i)=s*eye(l);
end
sed=0; 
[class2_X,class2_y]=mixt_model(m2,S2,P2,N2,sed);

% Form X1
X1=[class1_X  class2_X]; %Data vectors
y1=[ones(1,N1) -ones(1,N2)]; %Class labels
figure(1), hold on
figure(1), plot(X1(1,y1==1),X1(2,y1==1),'r.',X1(1,y1==-1),X1(2,y1==-1),'bx')

% Generate test set X2 %%%%%生成测试样本

% Data of the first class
sed=100; %Random generator seed. This time we set this value to 100
[class1_X,class1_y]=mixt_model(m1,S1,P1,N1,sed);

% Data of the second class
sed=100; %Random generator seed
[class2_X,class2_y]=mixt_model(m2,S2,P2,N2,sed);

%Production of the unified data set
X2=[class1_X class2_X]; %Data vectors
y2=[ones(1,N1) -ones(1,N2)]; %Class labels
%%%%%%生成数据部分结束

% 
%%%第二大步利用BP网络进行训练，利用两层网络，隐层用四个节点


rand('seed',100) 
randn('seed',100)
iter=9000; %Number of iterations

[net,tr]=NN_training(X1,y1,k,code,iter,par_vec);

% Compute the training and the test errors
pe_train=NN_evaluation(net,X1,y1)
pe_test=NN_evaluation(net,X2,y2)

% Plot the data points as well as the decision regions of the FNN
maxi=max(max([X1'; X2']));
mini=min(min([X1'; X2']));
bou=[mini maxi];
fig_num=2; % figure handle
resolu=(bou(2)-bou(1))/100; % figure resolution
plot_NN_reg(net,bou,resolu,fig_num); % plot decision region
figure(fig_num), hold on % plot training set
figure(fig_num), plot(X1(1,y1==1),X1(2,y1==1),'r.', X1(1,y1==-1),X1(2,y1==-1),'bx')

% Plot the training error versus the number of iterations
figure(3), plot(tr.perf)
