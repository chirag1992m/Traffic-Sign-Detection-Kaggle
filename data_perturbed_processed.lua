-- For Training Data
------PreProcess
-- This file will preload the data
-- Crop the image to the proper signal
-- Change to YUV color space
-- Scale the image to the given size
------On the fly
-- Add Jitter (rotation, transformation)
-- Gloabally normalize
-- And send to network

-- For test data
------Preprocess
-- Preload the data
-- Crop the image to the proper signal
-- Change to YUV color space
-- Scale the image to the given size
-- Globally normalize

local optParser = require 'opts'
local opt = optParser.parse(arg)

local image = require 'image'
local tnt = require 'torchnet'

local DATA_PATH = (opt.data ~= '' and opt.data or './data/')
local WIDTH, HEIGHT = opt.imageSize, opt.imageSize

local trainDataInfo = torch.load(DATA_PATH..'train.t7')
local testDataInfo = torch.load(DATA_PATH..'test.t7')
local valDataInfo = torch.load(DATA_PATH..'validation.t7')


-- Preloading the data
function pre_process(img)
    f = tnt.transform.compose {
        [1] = function (inp) return image.rgb2yuv(inp) end,
        [2] = function (inp) return image.scale(inp, WIDTH, HEIGHT) end
    }
    return f(img)
end

function load_and_crop_train (sample)
    classId, track, file = sample[9], sample[1], sample[2]
    file = DATA_PATH .. '/train_images/'.. string.format("%05d/%05d_%05d.ppm", classId, track, file)
    img = image.load(file)

    x1, y1, x2, y2 = sample[5], sample[6], sample[7], sample[8]
    return pre_process(image.crop(img, x1, y1, x2, y2))
end

function load_and_crop_test (sample)
    file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", sample[1])
    img = image.load(file)

    x1, y1, x2, y2 = sample[4], sample[5], sample[6], sample[7]
    return pre_process(image.crop(img, x1, y1, x2, y2))
end

channels = 3

local trainData = torch.Tensor(trainDataInfo:size(1), channels, WIDTH, HEIGHT)
local trainDataLabel = torch.LongTensor(trainDataInfo:size(1))
for i=1, trainDataInfo:size(1) do
    trainData[i] = load_and_crop_train(trainDataInfo[i])
    trainDataLabel[i] = trainDataInfo[i][9] + 1
end

local valData = torch.Tensor(valDataInfo:size(1), channels, WIDTH, HEIGHT)
local valDataLabel = torch.LongTensor(valDataInfo:size(1))
for i=1, valDataInfo:size(1) do
    valData[i] = load_and_crop_train(valDataInfo[i])
    valDataLabel[i] = valDataInfo[i][9] + 1
end

local testData = torch.Tensor(testDataInfo:size(1), channels, WIDTH, HEIGHT)
for i=1, testDataInfo:size(1) do
    testData[i] = load_and_crop_test(testDataInfo[i])
end
--Preloading complete


--Pre-calculate std deviation and mean for global normalization
local mean_global = torch.zeros(3)
local std_global = torch.zeros(3)

for i=1, channels do
    mean_global[i] = trainData[{{}, i, {}, {}}]:mean()
    std_global[i] = trainData[{{}, i, {}, {}}]:mean()
end


--Normalize test and validation data
for i=1, channels do
    testData[{{}, i, {}, {}}]:add(-mean_global[i])
    testData[{{}, i, {}, {}}]:div(std_global[i])

    valData[{{}, i, {}, {}}]:add(-mean_global[i])
    valData[{{}, i, {}, {}}]:div(std_global[i])
end

-- Function to handle Jittering and Preprocessing

function BrightnessJitter(inp)
    if torch.uniform() < 0.5 then
        return inp
    end
    jitter = 1.0 - (torch.uniform(-0.3, 0.3))
    inp:mul(jitter)
    return inp
end

function ContrastJitter(inp)
    if torch.uniform() < 0.5 then
        return inp
    end
    local contrast = torch.Tensor(inp:size())
    for i=1, inp:size()[1] do
        contrast[i]:fill(inp[i]:mean())
    end
    jitter = 1.0 - (torch.uniform(-0.3, 0.3))

    return inp:mul(jitter):add(1 - jitter, contrast)
end

function RandomRotate(inp)
    if torch.uniform() < 0.5 then
        return inp
    end
    return image.rotate(inp, (torch.uniform() - 0.5) * 15 * math.pi / 180, 'bilinear')
end

function RandomTranslate(inp)
    if torch.uniform() < 0.5 then
        return inp
    end
    jitter = torch.uniform(-2, 2)
    return image.translate(inp, jitter, jitter)
end


function AddJitter(inp)
    f = tnt.transform.compose{
--        ContrastJitter,
--        BrightnessJitter,
        RandomRotate,
        RandomTranslate
    }
    return f(inp)
end

function GlobalContrastNormalize(inp)
    for i=1, channels do
        inp[i]:add(-mean_global[i])
        inp[i]:div(-std_global[i])
    end
    return inp
end

function GetTrainSample(inp)
    f = tnt.transform.compose{
        AddJitter,
        GlobalContrastNormalize
    }
    return f(inp)
end

--Creating the datasets
local valDataset = tnt.ListDataset{
    list = torch.range(1, valData:size(1)):long(),
    load = function(idx)
        return {
            input = valData[idx],
            target = torch.LongTensor{valDataLabel[idx]}
        }
    end
}

local trainDataset = tnt.ShuffleDataset {
    dataset = tnt.ListDataset{
        list = torch.range(1, trainData:size(1)):long(),
        load = function(idx)
            return {
                input = GetTrainSample(trainData[idx]),
                target = torch.LongTensor{trainDataLabel[idx]}
            }
        end
    }
}

local testDataset = tnt.ListDataset{
    list = torch.range(1, testData:size(1)):long(),
    load = function(idx)
        return {
            input = testData[idx],
            target = torch.LongTensor{0}
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

M = {}

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

                    local channels = 3
                    local BrightnessJitter = function (inp)
                        if torch.uniform() < 0.5 then
                            return inp
                        end
                        jitter = 1.0 - (torch.uniform(-0.3, 0.3))
                        inp:mul(jitter)
                        return inp
                    end

                    local ContrastJitter = function (inp)
                        if torch.uniform() < 0.5 then
                            return inp
                        end
                        local contrast = torch.Tensor(inp:size())
                        for i=1, inp:size()[1] do
                            contrast[i]:fill(inp[i]:mean())
                        end
                        jitter = 1.0 - (torch.uniform(-0.3, 0.3))

                        return inp:mul(jitter):add(1 - jitter, contrast)
                    end

                    local RandomRotate = function (inp)
                        if torch.uniform() < 0.5 then
                            return inp
                        end
                        return image.rotate(inp, (torch.uniform() - 0.5) * 15 * math.pi / 180, 'bilinear')
                    end

                    local RandomTranslate = function (inp)
                        if torch.uniform() < 0.5 then
                            return inp
                        end
                        jitter = torch.uniform(-2, 2)
                        return image.translate(inp, jitter, jitter)
                    end


                    local AddJitter = function (inp)
                        f = tnt.transform.compose{
--                            ContrastJitter,
--                            BrightnessJitter,
                            RandomRotate,
                            RandomTranslate
                        }
                        return f(inp)
                    end

                    local GlobalContrastNormalize = function (inp)
                        for i=1, channels do
                            inp[i]:add(-mean_global[i])
                            inp[i]:div(-std_global[i])
                        end
                        return inp
                    end

                    local GetTrainSample = function (inp)
                        f = tnt.transform.compose{
                            AddJitter,
                            GlobalContrastNormalize
                        }
                        return f(inp)
                    end

                    return tnt.BatchDataset{
                        batchsize = opt.batchsize,
                        dataset = tnt.ShuffleDataset {
                            dataset = tnt.ListDataset{
                                list = torch.range(1, trainData:size(1)):long(),
                                load = function(idx)
                                    return {
                                        input = GetTrainSample(trainData[idx]),
                                        target = torch.LongTensor{trainDataLabel[idx]}
                                    }
                                end
                            }
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