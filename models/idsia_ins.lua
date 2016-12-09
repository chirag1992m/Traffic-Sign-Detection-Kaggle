-- 3 x 48 x 48 image

local nn = require 'nn'

local Convolution = nn.SpatialConvolution
local SNorm = nn.SpatialBatchNormalization
local RelU = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear
local Regularization = nn.Dropout

local model  = nn.Sequential()

model:add(Convolution(3, 100, 7, 7))
model:add(SNorm(100,1e-3))
model:add(RelU())
model:add(Max(2,2,2,2))
model:add(Convolution(100, 150, 4, 4))
model:add(SNorm(150,1e-3))
model:add(RelU())
model:add(Max(2,2,2,2))
model:add(Convolution(150, 250, 4, 4))
model:add(SNorm(250,1e-3))
model:add(RelU())
model:add(Max(2,2,2,2))
model:add(View(2250))
model:add(Linear(2250, 300))
model:add(RelU())
model:add(Regularization(0.5))
model:add(Linear(300, 43))

return model
