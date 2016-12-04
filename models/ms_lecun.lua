-- 3 x 32 x 32 image

local nn = require 'nn'

local Conv = nn.SpatialConvolution
local Non_linear = nn.Tanh
local Pool = nn.SpatialSubSampling
local Vector = nn.Reshape
local FC = nn.Linear
local Reg = nn.Dropout

local model  = nn.Sequential()

model:add(Conv(3, 108, 5, 5))
model:add(Pool(108, 2, 2, 2, 2))
model:add(Non_linear())

Concatenator = nn.Concat(2)

branch_1 = nn.Sequential()
branch_1:add(Conv(108, 108, 5, 5))
branch_1:add(Pool(108, 2, 2, 2, 2))
branch_1:add(Non_linear())
branch_1:add(Vector(2700))

branch_2 = nn.Sequential()
branch_2:add(Pool(108, 2, 2, 2, 2))
branch_2:add(Non_linear())
branch_2:add(Vector(5292))

Concatenator:add(branch_1)
Concatenator:add(branch_2)

model:add(Concatenator)

model:add(FC(7992, 100))
model:add(Non_linear())
model:add(Reg(0.5))
model:add(FC(100, 100))
model:add(Non_linear())
model:add(Reg(0.5))
model:add(FC(100, 43))

return model
