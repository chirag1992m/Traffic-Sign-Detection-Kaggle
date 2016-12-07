-- 3 x 32 x 32 image

local nn = require 'nn'


local Convolution = nn.SpatialConvolution
local Tanh = nn.Tanh
local View = nn.View
local Linear = nn.Linear
local Regularization = nn.Dropout

local model  = nn.Sequential()

model:add(Convolution(3, 16, 5, 5))
model:add(Tanh())
model:add(Convolution(16, 16, 2, 2, 2, 2))
model:add(Tanh())
model:add(Convolution(16, 128, 5, 5))
model:add(Tanh())
model:add(Convolution(128, 128, 2, 2, 2, 2))
model:add(Tanh())
model:add(View(3200))
model:add(Linear(3200, 64))
model:add(Regularization(0.5))
model:add(Tanh())
model:add(Linear(64, 43))

return model
