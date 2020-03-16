%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Demo code for the following paper:
%%%
%%% Ziming Zhang and Venkatesh Saligrama. "Zero-Shot Learning via Semantic
%%% Similarity Embedding". In ICCV, 2015.
%%%
%%% research purpose only. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dataset_path = 'datasets/aPascal-aYahoo';
load(sprintf('%s/class_attributes.mat', dataset_path));
load(sprintf('%s/cnn_feat_imagenet-vgg-verydeep-19.mat', dataset_path));

trainClassLabels = 1:20;
testClassLabels = 21:32;

%%  train
Templates = zeros(size(cnn_feat,1), length(trainClassLabels), 'single');
for i = 1:length(trainClassLabels)
    Templates(:,i) = mean(cnn_feat(:,labels==trainClassLabels(i)), 2);
end

%% source domain
A = class_attributes';

%%% linear kernel   (H + 1e-3)
A = A ./ repmat(sqrt(sum(A.^2, 1))+eps, [size(A,1) 1]);
A = A' * A;

%%% projection
H = A(trainClassLabels, trainClassLabels);
F = A(trainClassLabels, :);
alpha = zeros(length(trainClassLabels), size(F,2));

%%% qpas is a quadratic programming solver. You can change it to any QP solver you have (e.g. the default matlab QP solver)

for i = 1:size(F,2)
    f = -F(:,i);    
    alpha(:,i) = qpas(double(H + 1e1*eye(size(H,2))),double(f),[],[], ...
        ones(1,length(f)),1,zeros(length(f),1));    
end

%%  target domain
train_id = find(ismember(labels, trainClassLabels));
x = zeros(4096, length(trainClassLabels), length(trainClassLabels), 'single');
for i = 1:length(train_id)   
%     d = min(repmat(cnn_feat(:,train_id(i)), [1 size(Templates,2)]), Templates);    % intersection
    d = max(0, repmat(cnn_feat(:,train_id(i)), [1 size(Templates,2)])-Templates);    % ReLU
    x(:,:,labels(train_id(i))==trainClassLabels) = ...
        x(:,:,labels(train_id(i))==trainClassLabels) + single(d*alpha(:,trainClassLabels));     
end
y = [];
for i = 1:length(trainClassLabels)
    d = -ones(size(alpha,2), 1);
    d(trainClassLabels(i)) = 1;
    y = [y; d(trainClassLabels)];
end
x = reshape(x, size(x,1), []);

%%% train svms
%%% svmocas is an svm solver. You can change it to any svm solver you have (e.g. liblinear)

maxval = max(abs(x(:)));
rand_id = randsample(find(y==-1), 50);
[w b stat] = svmocas(x./maxval, 1, double(y), 1e1); 


%%% update models
for iter = 1:10
    
    %%% gradient
    grad = zeros(length(w), size(alpha,1));
    for i = 1:length(train_id)   
%         d = min(repmat(cnn_feat(:,train_id(i)), [1 size(Templates,2)]), Templates);    
        d = max(0, repmat(cnn_feat(:,train_id(i)), [1 size(Templates,2)])-Templates); 
        val = (w' * d) * alpha;
        y = -ones(1, size(alpha,2));
        y(labels(train_id(i))) = 1;
        dec = single(val.*y<0);
        if any(dec==1)
            grad = grad + (w*((dec.*y)*alpha'))...
                .* single(repmat(cnn_feat(:,train_id(i)), [1 size(alpha,1)])<Templates);     % gradient needs to be adjusted for intersection
        end
    end
%     Templates = max(0, Templates + 1e-2*grad./length(train_id));     % no l1: 1e0, with l1: 1e-3
    Templates = max(0, Templates - 1e-3*grad./length(train_id));
        

    %%% sample data
    x = zeros(4096, length(trainClassLabels), length(trainClassLabels), 'single');
    for i = 1:length(train_id)   
%         d = min(repmat(cnn_feat(:,train_id(i)), [1 size(Templates,2)]), Templates);    
        d = max(0, repmat(cnn_feat(:,train_id(i)), [1 size(Templates,2)])-Templates);   
        x(:,:,labels(train_id(i))==trainClassLabels) = x(:,:,labels(train_id(i))==trainClassLabels) + single(d*alpha(:,trainClassLabels));  
    end
    y = [];
    for i = 1:length(trainClassLabels)
        d = -ones(size(alpha,2), 1);
        d(trainClassLabels(i)) = 1;
        y = [y; d(trainClassLabels)];
    end
    x = reshape(x, size(x,1), []);

    %%% train svms
    maxval = max(abs(x(:)));
    [w b stat] = svmocas(x./maxval, 1, double(y(:)), 1e1);     
end

%%  test
test_id = find(ismember(labels, testClassLabels));
margins = [];
for i = 1:length(test_id)    
%     d = min(repmat(cnn_feat(:,test_id(i)), [1 size(Templates,2)]), Templates);
    d = max(0, repmat(cnn_feat(:,test_id(i)), [1 size(Templates,2)])-Templates); 
    d = w' * d * alpha(:,testClassLabels);
    margins = [margins; d];
end
%%% classify
[margin id] = max(margins, [], 2);
acc = sum(testClassLabels(id)'==labels(test_id))/length(test_id);

