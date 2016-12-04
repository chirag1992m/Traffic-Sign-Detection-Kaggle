-- 3 x 32 x 32 image

local nn = require 'nn'

local Conv = nn.SpatialConvolution
local Non_linear = nn.ReLU
local Pool = nn.SpatialMaxPooling
local Vector = nn.Reshape
local FC = nn.Linear

local model  = nn.Sequential()

model:add(Conv(3, 16, 5, 5))
model:add(Pool(2, 2, 2, 2))
model:add(Non_linear())

Concatenator = nn.Concat(2)

branch_1 = nn.Sequential()
branch_1:add(Conv(16, 128, 5, 5))
branch_1:add(Pool(2, 2, 2, 2))
branch_1:add(Non_linear())
branch_1:add(Vector(3200))

branch_2 = Vector(3136)

Concatenator:add(branch_1)
Concatenator:add(branch_2)

model:add(Concatenator)

model:add(FC(6336, 1000))
model:add(Non_linear())
model:add(FC(1000, 43))

return model
