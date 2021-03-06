local _ = require 'moses'
local classic = require 'classic'
local gnuplot = require 'gnuplot'
local Evaluator = require 'Evaluator'

local Validation = classic.class('Validation')
function Validation:_init(opt, agent, env, display, opt2, agent2)
  self.opt = opt
  self.opt2 = opt2
  self.agent = agent
  self.agent2 = agent2 or false
  self.env = env
  self.saveInd = 0
  --Set up Results log
  self.rFile = io.open(paths.concat(self.opt.experiments, self.opt._id, "Results.txt"), "w") -- Global results
  self.rFile:write("Total Score\t Average Score\t Averaged Q\t Delta\n")
  self.rFile:flush()
  if self.agent2 then
    self.rFile2 = io.open(paths.concat(self.opt.experiments, self.opt._id, "Results2.txt"), "w") -- Global results
    self.rFile2:write("Total Score\t Average Score\t Averaged Q\t Delta\n")
    self.rFile2:flush()
  end
  self.hasDisplay = false
  if display then
    self.hasDisplay = true
    self.display = display
  end

  -- Create (Atari normalised score) evaluator
  self.evaluator = Evaluator(opt.game)

  self.bestValScore = _.max(self.agent.valScores) or -math.huge -- Retrieve best validation score from agent if available

  self.evalEpisodes = opt.evalEpisodes

  classic.strict(self)
end


function Validation:validate()
  log.info('Validating')
  -- Set environment and agent to evaluation mode
  self.env:evaluate()
  self.agent:evaluate()
  if self.agent2 then
    self.agent2:evaluate()
  end

  -- Start new game
  local reward, state, terminal = 0, self.env:start(), false

  -- Validation variables
  local valEpisode = 1
  local valEpisodeScore = 0
  local valTotalScore = 0
  local valStepStrFormat = '%0' .. (math.floor(math.log10(self.opt.valSteps)) + 1) .. 'd' -- String format for padding step with zeros

  for valStep = 1, self.opt.valSteps do
    -- Observe and choose next action (index)
    local actionA = self.agent:observe(reward, state, terminal)
    local actionB = nil    
    if self.agent2 then
        actionB = self.agent2:observe(-reward, state, terminal)
    end
    
    if not terminal then
      -- Act on environment
      reward, state, terminal = self.env:step(actionA, actionB)
      -- Track score
      valEpisodeScore = valEpisodeScore + reward
    else
      -- Print score every 10 episodes
      if valEpisode % 10 == 0 then
        log.info('[VAL] Steps: ' .. string.format(valStepStrFormat, valStep) .. '/' .. self.opt.valSteps .. ' | Episode ' .. valEpisode .. ' | Score: ' .. valEpisodeScore)
      end

      -- Start a new episode
      valEpisode = valEpisode + 1
      reward, state, terminal = 0, self.env:start(), false
      valTotalScore = valTotalScore + valEpisodeScore -- Only add to total score at end of episode
      valEpisodeScore = reward -- Reset episode score
    end

    -- Display (if available)
    if self.hasDisplay then
      self.display:display(self.agent, self.env:getDisplay())
    end
  end

  -- If no episodes completed then use score from incomplete episode
  if valEpisode == 1 then
    valTotalScore = valEpisodeScore
  end  
  -- Print total and average score
  log.info('Total Score: ' .. valTotalScore)
  valTotalScore = valTotalScore/math.max(valEpisode - 1, 1) -- Only average score for completed episodes in general
  log.info('Average Score: ' .. valTotalScore)
  -- Pass to agent (for storage and plotting)
  self.agent.valScores[#self.agent.valScores + 1] = valTotalScore
  -- Calculate normalised score (if valid)
  local normScore = self.evaluator:normaliseScore(valTotalScore)
  if normScore then
    log.info('Normalised Score: ' .. normScore)
    self.agent.normScores[#self.agent.normScores + 1] = normScore
  end

  -- Visualise convolutional filters
  self.agent:visualiseFilters()

  -- Use transitions sampled for validation to test performance
  local avgV, avgTdErr = self.agent:validate()
  
  log.info('Average V: ' .. avgV)
  log.info('Average δ: ' .. avgTdErr)
  --print results
  self.rFile:write(valTotalScore .."\t "..valTotalScore.."\t "..avgV .."\t "..avgTdErr .."\n")
  self.rFile:flush()
  if self.agent2 then
    local avgV2, avgTdErr2 = self.agent2:validate()
    self.rFile2:write(-valTotalScore .."\t "..-valTotalScore.."\t "..avgV2 .."\t "..avgTdErr2 .."\n")
    self.rFile2:flush()
  end  
  -- Save latest weights
  log.info('Saving weights')
  self.agent:saveWeights(paths.concat(self.opt.experiments, self.opt._id, 'last.weights.t7'))
  --saving several weights
  log.info("Saved Network Index: " .. self.saveInd)
  local weightName = 'weights'..self.saveInd..'.t7'
  self.agent:saveWeights(paths.concat(self.opt.experiments, self.opt._id, weightName))
  if self.agent2 then
    self.agent2:saveWeights(paths.concat(self.opt.experiments, self.opt2._id, weightName))
  end
  self.saveInd=(self.saveInd+1) % self.opt.saveAgents
  -- Save "best weights" if best score achieved
  if valTotalScore > self.bestValScore then
    log.info('New best average score')
    self.bestValScore = valTotalScore

    log.info('Saving new best weights')
    self.agent:saveWeights(paths.concat(self.opt.experiments, self.opt._id, 'best.weights.t7'))
  end
  
  -- Set environment and agent to training mode
  self.env:training()
  self.agent:training()
  if self.agent2 then
    self.agent2:training()
  end
end


function Validation:evaluate()
  log.info('Evaluation mode')
  -- Set environment and agent to evaluation mode
  self.env:evaluate()
  self.agent:evaluate()
  if self.agent2 then
    self.agent2:evaluate()
  end
  
  local reward, state, terminal
  
  -- Report episode score
  local episodeScore

  -- Play evalEpisodes games (episodes)
  for episodeNum = 1, self.evalEpisodes do
    reward, state, terminal = 0, self.env:start(), false
  
    episodeScore = reward
    
    local actionA = nil
    local actionB = nil
  
    local step = 1
    while not terminal and (step<self.opt.evalSteps) do
      -- Observe and choose next action (index)
      actionA = self.agent:observe(reward, state, terminal)
      if self.agent2 then
        actionB = self.agent2:observe(-reward, state, terminal)
      else
        actionB = nil
      end
      -- Act on environment
      reward, state, terminal = self.env:step(actionA, actionB)
      episodeScore = episodeScore + reward
  
      -- Record (if available)
      if self.hasDisplay then
        self.display:display(self.agent, self.env:getDisplay(), step)
      end
      -- Increment evaluation step counter
      step = step + 1
    end
    log.info('Final Score, episode ' .. episodeNum ..': ' .. episodeScore)
  
    -- Record (if available)
    if self.hasDisplay and episodeNum == 1 then
      self.display:createVideo()
    end
  end
end


return Validation
