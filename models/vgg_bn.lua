require 'nn'

local Conv = nn.SpatialConvolution
local SNorm = nn.SpatialBatchNormalization
local Norm = nn.BatchNormalization
local NonLin = nn.ReLU
local Reg = nn.Dropout
local FC = nn.Linear
local SubSample = nn.SpatialMaxPooling

local model = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  model:add(Conv(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  model:add(SNorm(nOutputPlane,1e-3))
  model:add(NonLin(true))
  return model
end

ConvBNReLU(3,64)
ConvBNReLU(64,64)
model:add(SubSample(2,2,2,2):ceil())

ConvBNReLU(64,128)
ConvBNReLU(128,128)
model:add(SubSample(2,2,2,2):ceil())

ConvBNReLU(128,256)
ConvBNReLU(256,256)
ConvBNReLU(256,256)
model:add(SubSample(2,2,2,2):ceil())

ConvBNReLU(256,512)
ConvBNReLU(512,512)
ConvBNReLU(512,512)
model:add(SubSample(2,2,2,2):ceil())

ConvBNReLU(512,512)
ConvBNReLU(512,512)
ConvBNReLU(512,512)
model:add(SubSample(2,2,2,2):ceil())
model:add(nn.View(512))

model:add(Reg(0.5))
model:add(FC(512,512))
model:add(Norm(512))
model:add(NonLin(true))
model:add(Reg(0.5))
model:add(FC(512,43))

-- initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'nn.SpatialConvolution'
end

MSRinit(model)

return model