require 'torch'
require 'optim'
require 'os'
require 'xlua'

--[[
--  Hint:  Plot as much as you can.  
--  Look into torch wiki for packages that can help you plot.
--]]

local tnt = require 'torchnet'
local image = require 'image'
local optParser = require 'opts'
local opt = optParser.parse(arg)

if opt.cuda then
    require 'cunn'
    require 'cudnn' -- faster convolutions

    cudnn.benchmark = true
    cudnn.fastest = true
    cudnn.verbose = true
end

local WIDTH, HEIGHT = opt.imageSize, opt.imageSize
local DATA_PATH = (opt.data ~= '' and opt.data or './data/')

local file_suffix = opt.suffix .. string.format("_%d", os.time());
print("Training Started")

torch.setdefaulttensortype('torch.DoubleTensor')

torch.setnumthreads(opt.nThreads)
torch.manualSeed(opt.manualSeed)

if opt.cuda then
    cutorch.manualSeedAll(opt.manualSeed)
end

function resize(img)
    return image.scale(img, WIDTH,HEIGHT)
end

--[[
-- Hint:  Should we add some more transforms? shifting, scaling?
-- Should all images be of size 32x32?  Are we losing 
-- information by resizing bigger images to a smaller size?
--]]
function transformInput(inp)
    f = tnt.transform.compose{
        [1] = resize
    }
    return f(inp)
end

function getTrainSample(dataset, idx)
    r = dataset[idx]
    classId, track, file = r[9], r[1], r[2]
    file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
    return transformInput(image.load(DATA_PATH .. '/train_images/'..file))
end

function getTrainLabel(dataset, idx)
    return torch.LongTensor{dataset[idx][9] + 1}
end

function getTestSample(dataset, idx)
    r = dataset[idx]
    file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", r[1])
    return transformInput(image.load(file))
end

local trainData = torch.load(DATA_PATH..'train_full.t7')
local testData = torch.load(DATA_PATH..'test.t7')

trainDataset = tnt.SplitDataset{
    partitions = {train=(1 - (opt.val/100.0)), val=(opt.val/100.0)},
    initialpartition = 'train',
    --[[
    --  Hint:  Use a resampling strategy that keeps the 
    --  class distribution even during initial training epochs 
    --  and then slowly converges to the actual distribution 
    --  in later stages of training.
    --]]
    dataset = tnt.ShuffleDataset{
        dataset = tnt.ListDataset{
            list = torch.range(1, trainData:size(1)):long(),
            load = function(idx)
                return {
                    input =  getTrainSample(trainData, idx),
                    target = getTrainLabel(trainData, idx)
                }
            end
        }
    }
}

testDataset = tnt.ListDataset{
    list = torch.range(1, testData:size(1)):long(),
    load = function(idx)
        return {
            input = getTestSample(testData, idx),
            target = torch.LongTensor{testData[idx][1]}
        }
    end
}

local getIterator = function (dataset)
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = opt.batchsize,
            dataset = dataset
        }
    }
end

--[[
if opt.cuda then
    getIterator = function (dataset)
        return tnt.ParallelDatasetIterator{
            nthread = opt.nThreads,

            init = function ()
                    local tnt = require 'torchnet'
                end,

            closure = function ()
                    local image = require 'image'

                    local resize = function(img)
                        return image.scale(img, WIDTH,HEIGHT)
                    end

                    local getTrainSample = function (dataset, idx)
                        r = dataset[idx]
                        classId, track, file = r[9], r[1], r[2]
                        file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
                        return resize(image.load(DATA_PATH .. '/train_images/'..file))
                    end

                    local getTrainLabel = function (dataset, idx)
                        return torch.LongTensor{dataset[idx][9] + 1}
                    end

                    local getTestSample = function (dataset, idx)
                        r = dataset[idx]
                        file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", r[1])
                        return transformInput(image.load(file))
                    end

                    return tnt.BatchDataset{
                        batchsize = opt.batchsize,
                        dataset = dataset
                    }
                end
        }
    end
end
--]]


local model = require("models/".. opt.model)
local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
local criterion = nn.CrossEntropyCriterion()
local clerr = tnt.ClassErrorMeter{topk = {1}}
local timer = tnt.TimeMeter()
local batch = 1

local trainingLosses, trainingErrors = {}, {}
local intermediateTL, intermediateTE = {}, {}
local validationLosses, validationErrors = {}, {}
local intermediateVL, intermediateVE = {}, {}
local timeVals = {}

function metricCollectorReset()
    intermediateTL, intermediateTE = {}, {}
    intermediateVL, intermediateVE = {}, {}
end


if opt.cuda then
    model = model:cuda()
    criterion = criterion:cuda()
end

print('\nModel: '..tostring(model)..'\n')

engine.hooks.onStart = function(state)
    meter:reset()
    clerr:reset()
    timer:reset()
    metricCollectorReset()
    batch = 1
    if state.training then
        mode = 'Train'
    else
        mode = 'Val'
    end
end


if opt.cuda then
    local inputGPU = torch.CudaTensor()
    local targetGPU = torch.CudaTensor()
    
    engine.hooks.onSample = function(state)
        inputGPU:resize(state.sample.input:size() ):copy(state.sample.input)
        targetGPU:resize(state.sample.target:size()):copy(state.sample.target)
        state.sample.input  = inputGPU
        state.sample.target = targetGPU
    end
end

engine.hooks.onForwardCriterion = function(state)
    meter:add(state.criterion.output)
    clerr:add(state.network.output, state.sample.target)
    if opt.verbose == true then
        print(string.format("%s Batch: %d/%d; avg. loss: %2.4f; avg. error: %2.4f",
                mode, batch, state.iterator.dataset:size(), meter:value(), clerr:value{k = 1}))
    else
        xlua.progress(batch, state.iterator.dataset:size())
    end

    if mode == 'Train' then
        intermediateTL[batch] = meter:value()
        intermediateTE[batch] = clerr:value{k = 1}
    else
        intermediateVL[batch] = meter:value()
        intermediateVE[batch] = clerr:value{k = 1}
    end

    batch = batch + 1 -- batch increment has to happen here to work for train, val and test.
    timer:incUnit()
end

engine.hooks.onEnd = function(state)
    print(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f",
    mode, meter:value(), clerr:value{k = 1}, timer:value()))

    if mode == 'Train' then
        timeVals[epoch] = timer:value()
    end
end

local epoch = 1

while epoch <= opt.nEpochs do
    trainDataset:select('train')
    engine:train{
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset),
        optimMethod = optim.sgd,
        maxepoch = 1,
        config = {
            learningRate = opt.LR,
            momentum = opt.momentum
        }
    }

    trainingLosses[epoch] = intermediateTL
    trainingErrors[epoch] = intermediateTE

    trainDataset:select('val')
    engine:test{
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset)
    }

    validationLosses[epoch] = intermediateVL
    validationErrors[epoch] = intermediateVE

    print('Done with Epoch '..tostring(epoch))
    epoch = epoch + 1
end

local submission = assert(io.open(opt.submissionDir .. "/submission_" .. file_suffix .. ".csv", "w"))
submission:write("Filename,ClassId\n")
batch = 1

--[[
--  This piece of code creates the submission
--  file that has to be uploaded in kaggle.
--]]
engine.hooks.onForward = function(state)
    local fileNames  = state.sample.target
    local _, pred = state.network.output:max(2)
    pred = pred - 1
    for i = 1, pred:size(1) do
        submission:write(string.format("%05d,%d\n", fileNames[i][1], pred[i][1]))
    end

    if opt.verbose == true then
        print(string.format("%s Batch: %d/%d;", "test", batch, state.iterator.dataset:size()))
    else
        xlua.progress(batch, state.iterator.dataset:size())
    end

    batch = batch + 1
end

engine.hooks.onEnd = function(state)
    submission:close()
end

engine:test{
    network = model,
    iterator = getIterator(testDataset)
}

-- Dump the results in files
model:clearState()
torch.save(opt.logDir .. "/model_" .. file_suffix .. ".model", model)

torch.save(opt.logDir .. "/trainingErrors_" .. file_suffix .. ".log", torch.Tensor(trainingErrors))
torch.save(opt.logDir .. "/trainingLosses_" .. file_suffix .. ".log", torch.Tensor(trainingLosses))
torch.save(opt.logDir .. "/validationErrors_" .. file_suffix .. ".log", torch.Tensor(validationErrors))
torch.save(opt.logDir .. "/validationLosses_" .. file_suffix .. ".log", torch.Tensor(validationLosses))
torch.save(opt.logDir .. "/timers" .. file_suffix .. ".log", torch.Tensor(timeVals))