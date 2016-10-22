local Setup = require 'Setup'
local Master = require 'Master'
local AsyncMaster = require 'async/AsyncMaster'
local AsyncEvaluation = require 'async/AsyncEvaluation'



-- Parse options and perform setup
local setup = Setup(arg)
local opt = setup.opt

local setup2 = nil
local opt2 = nil

print(opt.twoPlayers)
if opt.twoPlayers then
  local f = assert(io.open(opt.playerTwoOptionsFile, "r"))
  local content = f:read("*a")
  local args2 = {}
  for word in content:gmatch("%S+") do table.insert(args2, word) end
  setup2 = Setup(args2)
  opt2 = setup2.opt
  f:close()
end

-- Start master experiment runner
if opt.async then
  if opt.mode == 'train' then
    local master = AsyncMaster(opt)
    master:start()
  elseif opt.mode == 'eval' then
    local eval = AsyncEvaluation(opt)
    eval:evaluate()
  end
else
  local master = Master(opt, opt2)

  if opt.mode == 'train' then
    master:train()
  elseif opt.mode == 'eval' then
    master:evaluate()
  end
end
