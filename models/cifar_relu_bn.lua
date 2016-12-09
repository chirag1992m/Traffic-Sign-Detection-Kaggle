-- 3 x 32 x 32 image

local nn = require 'nn'

local Convolution = nn.SpatialConvolution
local SNorm = nn.SpatialBatchNormalization
local RelU = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear
local Regularization = nn.Dropout

local model  = nn.Sequential()

model:add(Convolution(3, 16, 5, 5))
model:add(SNorm(16,1e-3))
model:add(RelU())
model:add(Max(2,2,2,2))
model:add(Convolution(16, 128, 5, 5))
model:add(SNorm(128,1e-3))
model:add(RelU())
model:add(Max(2,2,2,2))
model:add(View(3200))
model:add(Linear(3200, 64))
model:add(RelU())
model:add(Regularization(0.3))
model:add(Linear(64, 43))

return model
