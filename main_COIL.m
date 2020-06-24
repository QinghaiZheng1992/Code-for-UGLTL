clc;clear
rand('seed',100);
addpath('ClusteringMeasure')
addpath('code_coregspectral')
addpath('dataset')
addpath('LRR')
addpath('twist')
addpath('funs')
%% load dataset
dataset='COIL20_3VIEWS.mat';
load(dataset);
gt=Y;
X1=X1;
X2=X2;
X3=X3;
numClust=length(unique(Y));
num_views=3;
data{1}=double(X1);
data{2}=double(X2);
data{3}=double(X3);

tr_num=size(X1,1);
X1=data{1};
X2=data{2};
X3=data{3};

for v=1:3
    X{v}=data{v};
    [X{v}]=NormalizeData(X{v});
    %X{v} = zscore(X{v},1);
end
 
ratio=1;
sigma(1)=ratio*optSigma(X1);
sigma(2)=ratio*optSigma(X2);
sigma(3)=ratio*optSigma(X3);
gt=Y;
cls_num = numClust;

%% Construct the initial similarity matrix
K=[];
T=cell(1,num_views);
for j=1:num_views
    options.KernelType = 'Gaussian';
    options.t = sigma(j);
    K(:,:,j) = constructKernel(data{j},data{j},options);
    T{j}=K(:,:,j);
%     D=diag(sum(K(:,:,j),2));
%     L_rw=D^-1*K(:,:,j);
%     T{j}=L_rw;
end
T_tensor = cat(3, T{:,:});
t = T_tensor(:);
V = length(data); N = size(data{1},1); %sample number
all_dim=0;
for k=1:V
    all_dim=all_dim+size(X{k},2);
    Z{k} = zeros(N,N); %Z{2} = zeros(N,N);
%     W{k} = zeros(N,N);
end
Z_tensor = cat(3, Z{:,:});
% W_tensor = cat(3, W{:,:});
S_tensor=T_tensor;
sX = [N, N, V];
%
H = eye(tr_num)-1/tr_num*ones(tr_num);
% St = X*H*X';
% invSt = inv(St);

for ii=1:3
    dim=size(X{ii},2);
    invXTX{ii}=inv(10^(-6)*eye(dim,dim)+X{ii}'*H*X{ii});
end
% for ii=1:3
%     H = eye(N)-1/N*ones(N);
%     dim=size(X{ii},2);
%     St = X{ii}'*H*X{ii};
%     invXTX{ii}=inv(St);
% end

%% parameters setting
iter = 0;
tol = 1e-6;
max_iter=20;
d=8;
W_cat_matrix=zeros(all_dim,d);
alpha=10;
beta=50;
gamma=1;
tao=beta/alpha;
while iter < max_iter
    fprintf('----processing iter %d--------\n', iter+1);
    Zpre=Z_tensor;
    Spre=S_tensor;
    Wpre=W_cat_matrix;
    s=S_tensor(:);
    % update Z
    [z, objV] = wshrinkObj(s,tao,sX,0,3)   ;
    Z_tensor = reshape(z, sX);
    Z{1}=Z_tensor(:,:,1);
    Z{2}=Z_tensor(:,:,2);
    Z{3}=Z_tensor(:,:,3);
    % update W
    for ii=1:num_views
        tmp_S=S_tensor(:,:,ii);
        tmp_D=diag(sum(tmp_S));
        tmp_L=tmp_D-tmp_S;
        M = invXTX{ii}*(X{ii}'*tmp_L*X{ii});
%         tmp_W = eig1(M, d, 0, 0);
        [tmp_W,~] = eigs(M, d);
        W{ii} =tmp_W*diag(1./sqrt(diag(tmp_W'*tmp_W)));
    end
    W_cat_matrix = [W{1};W{2};W{3}];
    
    % update S
    for ii=1:num_views
        A=zeros(N);
        distx = L2_distance_1(W{ii}'*X{ii}',W{ii}'*X{ii}');
        [temp, idx] = sort(distx,2);
        tmp_Z=Z{ii};
        for jj=1:N
%             idxa0 = 1:N;
            dxi = distx(jj,:);
            ad = -(dxi-alpha*tmp_Z(jj,:))/(alpha+gamma);
            A(jj,:) = EProjSimplex_new(ad);
        end
        S_tensor(:,:,ii)=A;
    end
    % check convergence
    tp_rank=rank(Z{1});
    if tp_rank<1.2*numClust
        break;
    end
    iter = iter+1;
end

S = zeros(N,N);
for k=1:num_views
    S = S + Z{k};
end
S=(S+S')/2;

C = SpectralClustering(S,cls_num);
[A nmi avgent] = compute_nmi(gt,C);
ACC = Accuracy(C,double(gt));
[f,p,r] = compute_f(gt,C);
[AR,RI,MI,HI]=RandIndex(gt,C);
fprintf('tao=%f, F=%f, P=%f, R=%f, nmi score=%f, avgent=%f,  AR=%f, ACC=%f,\n',tao,f(1),p(1),r(1),nmi(1),avgent(1),AR(1),ACC(1));
