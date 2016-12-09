require 'torch'
require 'optim'
require 'os'

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

torch.setdefaulttensortype('torch.DoubleTensor')

local WIDTH, HEIGHT = opt.imageSize, opt.imageSize
local profiler = require("lib/profiler")
local model = require("models/".. opt.model)

--Create a random input of the given size
local randomInput = torch.rand(torch.LongStorage{1, 3, WIDTH, HEIGHT})

local total_ops, layer_ops, total_paras, layer_paras = profiler.profile(model, randomInput)

-- Compute per layer opt counts
print("Operations...")
for i, info in pairs(layer_ops) do
    local name = info['name']
    local ops = info['ops']

    print(string.format('%-32s%d %s, %.3f%%', name..':', ops, 'Operations', ((ops/total_ops) * 100)))
end

-- Compute per layer opt counts
print("Parameters...")
for i, info in pairs(layer_paras) do
    local name = info['name']
    local paras = info['paras']

    print(string.format('%-32s%d %s, %.3f%%', name..':', paras, 'Parameters', ((paras/total_paras) * 100)))
end