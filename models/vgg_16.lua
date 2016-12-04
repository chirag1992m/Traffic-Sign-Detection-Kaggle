-- 3 x 224 × 224 image

local nn = require 'nn'

local Conv = nn.SpatialConvolution
local non_linear = nn.ReLU
local Pool = nn.SpatialMaxPooling
local View = nn.View
local FC = nn.Linear
local Reg = nn.Dropout

local model  = nn.Sequential()

model:add(Conv(3, 64, 3, 3, 1, 1, 1, 1))
model:add(non_linear())
model:add(Conv(64, 64, 3, 3, 1, 1, 1, 1))
model:add(non_linear())
model:add(Pool(2, 2, 2, 2, 0, 0):ceil())
model:add(Conv(64, 128, 3, 3, 1, 1, 1, 1))
model:add(non_linear())
model:add(Conv(128, 128, 3, 3, 1, 1, 1, 1))
model:add(non_linear())
model:add(Pool(2, 2, 2, 2, 0, 0):ceil())
model:add(Conv(128, 256, 3, 3, 1, 1, 1, 1))
model:add(non_linear())
model:add(Conv(256, 256, 3, 3, 1, 1, 1, 1))
model:add(non_linear())
model:add(Conv(256, 256, 3, 3, 1, 1, 1, 1))
model:add(non_linear())
model:add(Pool(2, 2, 2, 2, 0, 0):ceil())
model:add(Conv(256, 512, 3, 3, 1, 1, 1, 1))
model:add(non_linear())
model:add(Conv(512, 512, 3, 3, 1, 1, 1, 1))
model:add(non_linear())
model:add(Conv(512, 512, 3, 3, 1, 1, 1, 1))
model:add(non_linear())
model:add(Pool(2, 2, 2, 2, 0, 0):ceil())
model:add(Conv(512, 512, 3, 3, 1, 1, 1, 1))
model:add(non_linear())
model:add(Conv(512, 512, 3, 3, 1, 1, 1, 1))
model:add(non_linear())
model:add(Conv(512, 512, 3, 3, 1, 1, 1, 1))
model:add(non_linear())
model:add(Pool(2, 2, 2, 2, 0, 0):ceil())
model:add(View(-1):setNumInputDims(3))
model:add(FC(25088, 4096))
model:add(non_linear())
model:add(Reg(0.5))
model:add(FC(4096, 4096))
model:add(non_linear())
model:add(Reg(0.5))
model:add(FC(4096, 43))

return model