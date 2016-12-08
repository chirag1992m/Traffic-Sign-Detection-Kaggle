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

if opt.dontValidate then
    trainDataInfo = trainDataInfo:cat(valDataInfo, 1)
end


-- Preloading the data
print("PreProcessing data...")
function pre_process(img)
    f = tnt.transform.compose {
        function (inp) return image.rgb2yuv(inp) end,
        function (inp) return image.scale(inp, WIDTH, HEIGHT) end
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
local actualClassDistribution = torch.DoubleTensor(43):fill(0) -- 43 is the number of classes
local trainDataIndexes = {}
for i=1, trainDataInfo:size(1) do
    trainData[i] = load_and_crop_train(trainDataInfo[i])
    trainDataLabel[i] = trainDataInfo[i][9] + 1
    actualClassDistribution[trainDataLabel[i]] = actualClassDistribution[trainDataLabel[i]] + 1

    if trainDataIndexes[trainDataLabel[i]] == nil then
        trainDataIndexes[trainDataLabel[i]] = {}
    end
    trainDataIndexes[trainDataLabel[i]][#trainDataIndexes[trainDataLabel[i]] + 1] = i
end
local initialClassDistribution = torch.DoubleTensor(43):fill(actualClassDistribution:max()) -- Resample to 1:1 ratio
local currentClassDistribution = initialClassDistribution:clone()

--Upscale the trainDataIndexes to currentClassDistribtion
for i=1, 43 do
    local sample_size = #trainDataIndexes[i]
    for j=sample_size+1, currentClassDistribution[i] do
        index = torch.ceil(torch.uniform(1, sample_size))
        trainDataIndexes[i][j] = trainDataIndexes[i][index]
    end
end

local distributionParameter = 1.0
local distributionGradient = opt.resampler
local trainAuxilliary = nil

-- Construct the other datasets
local valData = torch.Tensor(valDataInfo:size(1), channels, WIDTH, HEIGHT)
local valDataLabel = torch.LongTensor(valDataInfo:size(1))
for i=1, valDataInfo:size(1) do
    valData[i] = load_and_crop_train(valDataInfo[i])
    valDataLabel[i] = valDataInfo[i][9] + 1
end

local testData = torch.Tensor(testDataInfo:size(1), channels, WIDTH, HEIGHT)
local testDataLabel = torch.Tensor(testDataInfo:size(1))
for i=1, testDataInfo:size(1) do
    testData[i] = load_and_crop_test(testDataInfo[i])
    testDataLabel[i] = testDataInfo[i][1]
end
print("PreProcessing done!")
--Preloading complete


--Pre-calculate std deviation and mean for global normalization
local mean_global = torch.zeros(3)
local std_global = torch.zeros(3)

for i=1, channels do
    mean_global[i] = trainData[{{}, i, {}, {}}]:mean()
    std_global[i] = trainData[{{}, i, {}, {}}]:std()
end

print('mean: '.. tostring(mean_global))
print('std: '.. tostring(std_global))

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
    inp = inp:clone()
    jitter = 1.0 - (torch.uniform(-0.3, 0.3))
    inp:mul(jitter)
    return inp
end

function ContrastJitter(inp)
    if torch.uniform() < 0.5 then
        return inp
    end
    inp = inp:clone()
    local contrast = torch.Tensor(inp:size())
    for i=1, inp:size()[1] do
        contrast[i]:fill(inp[i]:mean())
    end
    jitter = 1.0 - (torch.uniform(-0.3, 0.3))

    return inp:mul(jitter):add(1 - jitter, contrast)
end

function SaturationJitter(inp)
    if torch.uniform() < 0.5 then
        return inp
    end
    inp = inp:clone()
    local saturation = torch.Tensor(inp:size())
    saturation[1]:zero()
    saturation[1]:add(0.299, inp[1]):add(0.587, inp[2]):add(0.114, inp[3])
    saturation[2]:copy(saturation[1])
    saturation[3]:copy(saturation[1])
    jitter = 1.0 - (torch.uniform(-0.3, 0.3))

    return inp:mul(jitter):add(1 - jitter, saturation)
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
    jitter = torch.uniform(-3, 3)
    return image.translate(inp, jitter, jitter)
end


function AddJitter(inp)
    f = tnt.transform.compose{
        ContrastJitter,
        BrightnessJitter,
        SaturationJitter,
        RandomRotate,
        RandomTranslate
    }
    return f(inp)
end

function GlobalContrastNormalize(inp)
    inp = inp:clone()
    for i=1, channels do
        inp[{i, {}, {}}]:add(-mean_global[i])
        inp[{i, {}, {}}]:div(std_global[i])
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

function GetTestSample(inp)
    return inp
end

--Creating the datasets
local valDataset = tnt.ListDataset{
    list = torch.range(1, valData:size(1)):long(),
    load = function(idx)
        return {
            input = GetTestSample(valData[idx]),
            target = torch.LongTensor{valDataLabel[idx]}
        }
    end
}

local testDataset = tnt.ListDataset{
    list = torch.range(1, testData:size(1)):long(),
    load = function(idx)
        return {
            input = GetTestSample(testData[idx]),
            target = torch.LongTensor{testDataLabel[idx]}
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

function M.update(temp_epoch)
    currentClassDistribution = ((initialClassDistribution * distributionParameter) + ((1 - distributionParameter) * actualClassDistribution)):clone()
    currentClassDistribution:ceil()

    distributionParameter = distributionParameter * distributionGradient

    local total_size = currentClassDistribution:sum()
    trainAuxilliary = torch.LongTensor(total_size):fill(0)
    local k = 1
    for i=1, 43 do
        local size_difference = #trainDataIndexes[i] - currentClassDistribution[i]
        print("size_difference", size_difference)
        if size_difference > 0 then
            for j=1, size_difference do
                table.remove(trainDataIndexes[i])
            end
        else
            size_difference = size_difference * -1
            local sample_size = #trainDataIndexes[i]
            for j=1, size_difference do
                index = torch.ceil(torch.uniform(1, sample_size))
                trainDataIndexes[i][#trainDataIndexes[i] + 1] = trainDataIndexes[i][index]
            end
        end

        for j=1, #trainDataIndexes[i] do
            trainAuxilliary[k] = trainDataIndexes[i][j]
            k = k+1
        end
    end
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
                        inp = inp:clone()
                        jitter = 1.0 - (torch.uniform(-0.3, 0.3))
                        inp:mul(jitter)
                        return inp
                    end

                    local ContrastJitter = function (inp)
                        if torch.uniform() < 0.5 then
                            return inp
                        end
                        inp = inp:clone()
                        local contrast = torch.Tensor(inp:size())
                        for i=1, inp:size()[1] do
                            contrast[i]:fill(inp[i]:mean())
                        end
                        jitter = 1.0 - (torch.uniform(-0.3, 0.3))

                        return inp:mul(jitter):add(1 - jitter, contrast)
                    end

                    local SaturationJitter = function (inp)
                        if torch.uniform() < 0.5 then
                            return inp
                        end
                        inp = inp:clone()
                        local saturation = torch.Tensor(inp:size())
                        saturation[1]:zero()
                        saturation[1]:add(0.299, inp[1]):add(0.587, inp[2]):add(0.114, inp[3])
                        saturation[2]:copy(saturation[1])
                        saturation[3]:copy(saturation[1])
                        jitter = 1.0 - (torch.uniform(-0.3, 0.3))

                        return inp:mul(jitter):add(1 - jitter, saturation)
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
                            ContrastJitter,
                            BrightnessJitter,
                            SaturationJitter,
                            RandomRotate,
                            RandomTranslate
                        }
                        return f(inp)
                    end

                    local GlobalContrastNormalize = function (inp)
                        inp = inp:clone()
                        for i=1, channels do
                            inp[{i, {}, {}}]:add(-mean_global[i])
                            inp[{i, {}, {}}]:div(std_global[i])
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
                        dataset = tnt.ResampleDataset {
                            dataset = tnt.ShuffleDataset {
                                dataset = tnt.ListDataset {
                                        list = torch.range(1, trainData:size(1)):long(),
                                        load = function(idx)
                                            return {
                                                input = GetTrainSample(trainData[idx]),
                                                target = torch.LongTensor{trainDataLabel[idx]}
                                            }
                                        end
                                },
                            },

                            sampler = function (dataset, idx)
                                return trainAuxilliary[idx]
                            end,

                            size = trainAuxilliary:size(1)
                        }
                    }
                end
        }
    end
else
    M.getTrainIterator = function ()
        local trainDataset = tnt.ResampleDataset {
            dataset = tnt.ShuffleDataset {
                dataset = tnt.ListDataset{
                        list = torch.range(1, trainData:size(1)):long(),
                        load = function(idx)
                            return {
                                input = GetTrainSample(trainData[idx]),
                                target = torch.LongTensor{trainDataLabel[idx]}
                            }
                        end
                },
            },

            sampler = function (dataset, idx)
                return trainAuxilliary[idx]
            end,

            size = trainAuxilliary:size(1)
        }

        return getBatchIterator(trainDataset)
    end
end

return M