function [net, info] = cnn_mnist(varargin)
%CNN_MNIST  Demonstrates MatConvNet on MNIST

% run(fullfile(fileparts(mfilename('fullpath')),'..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.batchNormalization = true ;
opts.network = [] ;
opts.networkType = 'simplenn' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = 'train_process_record';
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if isempty(opts.network)
  net = cnn_mnist_init('batchNormalization', opts.batchNormalization, ...
    'networkType', opts.networkType);
else
  net = opts.network;
  opts.network = [];
end

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath);
else
  imdb = imgDataPreprocess();
  mkdir(opts.expDir);
  save(opts.imdbPath, '-struct', 'imdb');
end

net.meta.classes.name = ["good","bad"];

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainfn = @cnn_train;
  case 'dagnn', trainfn = @cnn_train_dag;
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3));

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;

