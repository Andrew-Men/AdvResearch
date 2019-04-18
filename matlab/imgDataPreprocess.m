function imdb = imgDataPreprocess()
% Preprocess the data
datasetSize = [72 68];
addError = 1;
errorRatio = 0.3;

data = zeros(50,50,1,sum(datasetSize));

for j = [0, 1]
    for i = 1:datasetSize(j+1)
        newGrayImgMatrix = rgb2gray(imread(fullfile('dataset01',int2str(j),sprintf('%02d.png', 1))));
        data(:,:,i+j*datasetSize(1)) = newGrayImgMatrix;
    end
end
dataMean = mean(data(:,:,:,[1:20,28:47]),4);
data = bsxfun(@minus, data, dataMean) ;

if addError
    for i = 1:length(data)
        data(:,:,1,i) = data(:,:,1,i) + errorRatio*mean(mean(dataMean))*rand(50);
    end
end

imdb.images.data = single(data);
imdb.images.data_mean = single(dataMean);
imdb.images.labels = [zeros(1,datasetSize(1)) ones(1,datasetSize(2))];
imdb.images.set = [ones(1,60),3*ones(1,12),ones(1,60),3*ones(1,8)];
imdb.meta.sets = {'train', 'val', 'test'};
imdb.meta.classes = {'0', '1'};

save('imdb.mat','-struct','imdb')
end


