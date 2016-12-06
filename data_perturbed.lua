local M = {}

local optParser = require 'opts'
local opt = optParser.parse(arg)

local image = require 'image'
local tnt = require 'torchnet'

local DATA_PATH = (opt.data ~= '' and opt.data or './data/')
local WIDTH, HEIGHT = opt.imageSize, opt.imageSize

local gen = torch.Generator()

function random_rotation(img)
    local rotation_degree = torch.random(gen, 0, 45)
    return image.rotate(img, rotation_degree/360.0)
end

function random_translations(img)
    local transaltion_x = torch.random(gen, 0, 5)
    local transaltion_y = torch.random(gen, 0, 5)

    return image.translate(img, transaltion_x, transaltion_y)
end

function augmentImage(inp)
    f = tnt.transform.compose{
        [1] = random_rotation,
        [2] = random_translations
    }
    return f(inp)
end

function resize(img)
    return image.scale(img, WIDTH,HEIGHT)
end

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
    return transformInput(augmentImage(image.load(DATA_PATH .. '/train_images/'..file)))
end

function getTrainLabel(dataset, idx)
    return torch.LongTensor{dataset[idx][9] + 1}
end

function getTestSample(dataset, idx)
    r = dataset[idx]
    file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", r[1])
    return transformInput(image.load(file))
end

local trainingData = torch.load(DATA_PATH..'train.t7')
local testData = torch.load(DATA_PATH..'test.t7')
local validationData = torch.load(DATA_PATH..'validation.t7')

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

if opt.dontValidate then
    trainingData = trainingData:cat(validationData, 1)
end

local trainDataset = tnt.ShuffleDataset {
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

                    local random_rotation = function (img)
                        local rotation_degree = torch.random(torch.Generator(), 0, 45)
                        return image.rotate(img, rotation_degree/360.0)
                    end

                    local random_translations = function (img)
                        local transaltion_x = torch.random(torch.Generator(), 0, 5)
                        local transaltion_y = torch.random(torch.Generator(), 0, 5)

                        return image.translate(img, transaltion_x, transaltion_y)
                    end

                    local augmentImage = function (inp)
                        f = tnt.transform.compose{
                            [1] = random_rotation,
                            [2] = random_translations
                        }
                        return f(inp)
                    end

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
                        return transformInput(augmentImage(image.load(DATA_PATH .. '/train_images/'..file)))
                    end

                    local getTrainLabel = function (dataset, idx)
                        return torch.LongTensor{dataset[idx][9] + 1}
                    end

                    return tnt.BatchDataset{
                        batchsize = opt.batchsize,
                        dataset = tnt.ShuffleDataset {
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