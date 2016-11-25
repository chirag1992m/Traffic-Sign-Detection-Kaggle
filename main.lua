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

local WIDTH, HEIGHT = 32, 32
local DATA_PATH = (opt.data ~= '' and opt.data or './data/')

local logFile = assert(io.open(string.format(opt.logDir .. "/logs_%d.log", os.time()), "w"))
logFile:write("Training Started \n")

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

local trainData = torch.load(DATA_PATH..'train.t7')
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

if opt.cuda then
    model = model:cuda()
    criterion = criterion:cuda()
end

logFile:write('\nModel: '..tostring(model)..'\n\n')

engine.hooks.onStart = function(state)
    meter:reset()
    clerr:reset()
    timer:reset()
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
        logFile:write(string.format("%s Batch: %d/%d; avg. loss: %2.4f; avg. error: %2.4f \n",
                mode, batch, state.iterator.dataset:size(), meter:value(), clerr:value{k = 1}))
    else
        xlua.progress(batch, state.iterator.dataset:size())
    end
    batch = batch + 1 -- batch increment has to happen here to work for train, val and test.
    timer:incUnit()
end

engine.hooks.onEnd = function(state)
    logFile:write(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f \n",
    mode, meter:value(), clerr:value{k = 1}, timer:value()))
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

    trainDataset:select('val')
    engine:test{
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset)
    }

    logFile:write('Done with Epoch '..tostring(epoch)..'\n')
    epoch = epoch + 1
end

local submission = assert(io.open(string.format(opt.submissionDir .. "/submission_%d.csv", os.time()), "w"))
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
        logFile:write(string.format("%s Batch: %d/%d; \n", "test", batch, state.iterator.dataset:size()))
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

logFile:write("The End!")
logFile:close()

-- Dump the results in files
model:clearState()
torch.save(string.format(opt.logDir .. "/model_%d.model", os.time()), model)
