local M = {}

local optParser = require 'opts'
local opt = optParser.parse(arg)

local image = require 'image'
local tnt = require 'torchnet'

local DATA_PATH = (opt.data ~= '' and opt.data or './data/')
local WIDTH, HEIGHT = opt.imageSize, opt.imageSize

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

local trainSize = trainData:size()[1]
local trainSplit = (torch.DoubleTensor{(1 - (opt.val/100.0))} * trainSize):floor()[1]

local perm = torch.randperm(trainSize)

local trainingData = trainData[perm[1]]:resize(1, 9)
local start = 2
for i=start, trainSplit do
    trainingData = trainingData:cat(trainData[perm[i]]:resize(1, 9), 1)
end

local validationData = trainData[perm[trainSplit + 1]]:resize(1, 9)
start = trainSplit + 2
for i=start, trainSize do
    validationData = validationData:cat(trainData[perm[i]]:resize(1, 9), 1)
end

--Creating the datasets
local valDataset = tnt.ListDataset{
    list = torch.range(1, validationData:size(1)):long(),
    load = function(idx)
        return {
            input = getTrainSample(validationData, idx),
            target = getTrainLabel(validationData, idx)
        }
    end
}

local trainDataset = tnt.ListDataset{
    list = torch.range(1, trainingData:size(1)):long(),
    load = function(idx)
        return {
            input = getTrainSample(trainingData, idx),
            target = getTrainLabel(trainingData, idx)
        }
    end
}

local testDataset = tnt.ListDataset{
    list = torch.range(1, testData:size(1)):long(),
    load = function(idx)
        return {
            input = getTestSample(testData, idx),
            target = torch.LongTensor{testData[idx][1]}
        }
    end
}

function getBatchIterator(dataset_to_iterate)
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = opt.batchsize,
            dataset = dataset_to_iterate
        }
    }
end

function M.getTestIterator()
    return getBatchIterator(testDataset)
end

function M.getValIterator()
    return getBatchIterator(valDataset)
end

if opt.cuda then
    M.getTrainIterator = function ()
        return tnt.ParallelDatasetIterator{
            nthread = opt.nThreads,

            init = function ()
                    local tnt = require 'torchnet'
                end,

            closure = function ()
                    local image = require 'image'

                    local resize = function(img)
                        return image.scale(img, WIDTH, HEIGHT)
                    end

                    local transformInput = function (inp)
                        f = tnt.transform.compose{
                            [1] = resize
                        }
                        return f(inp)
                    end

                    local getTrainSample = function (dataset, idx)
                        r = dataset[idx]
                        classId, track, file = r[9], r[1], r[2]
                        file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
                        return transformInput(image.load(DATA_PATH .. '/train_images/'..file))
                    end

                    local getTrainLabel = function (dataset, idx)
                        return torch.LongTensor{dataset[idx][9] + 1}
                    end

                    return tnt.BatchDataset{
                        batchsize = opt.batchsize,
                        dataset = tnt.ListDataset{
                            list = torch.range(1, trainingData:size(1)):long(),
                            load = function(idx)
                                return {
                                    input = getTrainSample(trainingData, idx),
                                    target = getTrainLabel(trainingData, idx)
                                }
                            end
                        }
                    }
                end
        }
    end
else
    M.getTrainIterator = function ()
        return getBatchIterator(trainDataset)
    end
end

return M