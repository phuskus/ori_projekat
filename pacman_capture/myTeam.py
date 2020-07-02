# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import math
from util import nearestPoint
from capture import COLLISION_TOLERANCE, manhattanDistance


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveStegnutiAgent', second='DefensiveStegnutiAgent', numTraining=1):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########
weightsOffensive = {'carryingFood': 15.814672361697648, 'danger': -19.253369171930068, 'collectedFood': 58.61966962071899, 'score': 42.804997259021086, 'dToFood': -4.724655824899629}

weightsDeffensive = {'nearInvader': 6.065403234386981, 'lostFood': -6.168767332377552, 'score': 18.510301823143802}

dirToVec = {
    "North": (0.0, 1.0),
    "South": (0.0, -1.0),
    "East": (1.0, 0.0),
    "West": (-1.0, 0.0),
    "Stop": (0.0, 0.0)
}

class StegnutiAgent(CaptureAgent):

    def __init__(self, index):
        self.weights = util.Counter()
        self.override = False
        self.alpha = 0.1
        self.minAlpha = 0.01
        self.gamma = 0.2
        self.gameNumber = 0
        self.trainingGames = 200
        self.alphaDecay = (self.alpha - self.minAlpha) / self.trainingGames
        print("Alpha decay", self.alphaDecay)
        CaptureAgent.__init__(self, index)

    def getWeights(self):
        if self.override:
            return self.overrideWeights
        else:
            return self.weights

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        self.mazeWidth = gameState.getWalls().width
        self.mazeHeight = gameState.getWalls().height
        CaptureAgent.registerInitialState(self, gameState)
        print("Epsilon:", self.getEpsilon())
        print("Alpha:", self.alpha)

    def getEpsilon(self):
        if self.override:
            return 0.0
        return (-1.0 / self.trainingGames)*self.gameNumber + 1

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        There's a epsilon chance of the agent choosing a random action.
        """
        actions = gameState.getLegalActions(self.index)
        roll = random.random()
        if roll < self.getEpsilon():
            chosenAction = random.choice(actions)
            self.learn(gameState, chosenAction)
            return chosenAction

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = []
        qVal = 0
        # print("ACTIONS")
        for a in actions:
            qVal = self.getQValue(gameState, a)
            # print("Action %s: %d" % (str(a), qVal))
            values.append(qVal)
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        chosenAction = random.choice(bestActions)
        #print(chosenAction)
        self.learn(gameState, chosenAction)

        return chosenAction

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getFeatures(self, state, action):
        raise NotImplementedError()

    def learn(self, state, action):
        raise NotImplementedError()

    def getQValue(self, state, action):
        """
        Computes a linear combination of features and feature weights
        """
        val = self.getFeatures(state, action) * self.getWeights()
        #print("Q VALUE")
        #print(val)
        return val


    def getStateValue(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.
        """
        possibleStateQValues = util.Counter()
        for action in state.getLegalActions(self.index):
            possibleStateQValues[action] = self.getQValue(state, action)

        if len(possibleStateQValues) > 0:
            return possibleStateQValues[possibleStateQValues.argMax()]
        return 0.0


    def updateWeights(self, state, action, nextState, reward, features):
        diff = self.alpha * ((reward + self.gamma * self.getStateValue(nextState)) - self.getQValue(state, action))
        for feature in features.keys():
            self.weights[feature] = self.weights[feature] + diff * features[feature]


    def final(self, gameState):
        print("Weights", self.index, ":")
        print(self.getWeights())
        self.gameNumber += 1
        self.alpha -= self.alphaDecay
        self.alpha = max(self.alpha, self.minAlpha)
        CaptureAgent.final(self, gameState)


class OffensiveStegnutiAgent(StegnutiAgent):
    def registerInitialState(self, gameState):
        StegnutiAgent.registerInitialState(self, gameState)
        self.timeElapsed = 0
        self.override = True
        self.overrideWeights = weightsOffensive

    def getNearestFoodDistance(self, state):
        minDist = 100.0
        myPos = state.getAgentState(self.index).getPosition()
        dist = 0.0
        for food in self.getFood(state).asList():
            dist = self.getMazeDistance(myPos, food)
            if dist < minDist:
                minDist = dist
        return minDist / 50.0

    def getFeatures(self, state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        nextState = self.getSuccessor(state, action)
        agentState = nextState.getAgentState(self.index)
        myPos = agentState.getPosition()

        # 1. Carrying food
        features['carryingFood'] = agentState.numCarrying / 20.0

        # 2. Distance from nearest defending ghost
        nearestDefender = 50.0
        dist = 0.0
        for i in self.getOpponents(nextState):
            a = nextState.getAgentState(i)
            if not a.isPacman and a.getPosition() is not None:
                dist = self.getMazeDistance(myPos, a.getPosition())
                if dist < nearestDefender:
                    nearestDefender = dist

        nearestDefender = 1.0 - nearestDefender / 50.0
        # Close to zero when far, very steep rize when very close

        features['danger'] = 0.000000001 * (math.e ** (20.7 * nearestDefender))

        # 3. Total food collected from enemy
        features['collectedFood'] = 1.0 - len(self.getFood(nextState).asList()) / 20.0

        # 4. Total score
        features['score'] = self.getScore(nextState) / 20.0

        # 5. Nearest food
        features['dToFood'] = self.getNearestFoodDistance(nextState)

        #print(features)
        return features


    def learn(self, state, action):
        # Get next state
        nextState = self.getSuccessor(state, action)
        agentState = nextState.getAgentState(self.index)
        features = self.getFeatures(state, action)

        # Calculate reward

        # Losing is VERY bad
        if nextState.isLost():
            print("-1000 for loss")
            return -1000

        # Winning is VERY good
        if nextState.isOver():
            print("1000 for victory")
            return 1000.0

        reward = 0.0

        # Eating food is good
        # reward += (20 - len(self.getFood(nextState).asList())) * 5.0

        # Returning food is good
        reward += nextState.getAgentState(self.index).numReturned * 10.0

        # Wasting time is bad
        timePenalty = self.timeElapsed * 0.05
        reward -= timePenalty
        self.timeElapsed += 1

        # Going near food is good
        reward += (1.0 - features['dToFood']) * 5.0

        # Going near defenders is bad
        # reward -= features['danger'] * 2.0

        # Colliding with ghosts is bad
        myFuturePos = agentState.getPosition()
        myPos = state.getAgentState(self.index).getPosition()
        for index in self.getOpponents(state):
            a = state.getAgentState(index)
            if a.isPacman:
                continue
            enemyPos = a.getPosition()
            dir = dirToVec[a.getDirection()]
            futureEnemyPos = (enemyPos[0] + dir[0], enemyPos[1] + dir[1])
            if myFuturePos == futureEnemyPos:
                print("Off: Ghosts suck")
                reward -= 10.0

        # Eating one pellet of food is good
        blueFood = state.getBlueFood()
        if blueFood[int(myFuturePos[0])][int(myFuturePos[1])]:
            print("Off: Food rules")
            reward += 100.0

        #print(self.timeElapsed * 0.05)
        #print("Rewarded agent", self.index, reward)
        #print(reward)
        self.updateWeights(state, action, nextState, reward, features)

    def final(self, gameState):
        print("\nWeights offensive")
        print(self.getWeights())
        print()
        self.gameNumber += 1
        self.alpha -= self.alphaDecay
        self.alpha = max(self.alpha, self.minAlpha)
        CaptureAgent.final(self, gameState)

class DefensiveStegnutiAgent(StegnutiAgent):
    def registerInitialState(self, gameState):
        StegnutiAgent.registerInitialState(self, gameState)
        self.override = True
        self.overrideWeights = weightsDeffensive

    def getNearestFoodDistance(self, state):
        minDist = 100.0
        myPos = state.getAgentState(self.index).getPosition()
        dist = 0.0
        for food in self.getFoodYouAreDefending(state).asList():
            dist = self.getMazeDistance(myPos, food)
            if dist < minDist:
                minDist = dist
        return minDist / 50.0

    def getFeatures(self, state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        nextState = self.getSuccessor(state, action)
        agentState = nextState.getAgentState(self.index)
        #print(state.getAgentState(self.index).getPosition())
        myPos = agentState.getPosition()

        # 1. Distance from nearest invading ghost
        nearestInvader = 50.0
        dist = 0.0
        for i in self.getOpponents(nextState):
            a = nextState.getAgentState(i)
            if a.isPacman and a.getPosition() is not None:
                dist = self.getMazeDistance(myPos, a.getPosition())
                if dist < nearestInvader:
                    nearestInvader = dist

        features['nearInvader'] = 1.0 - nearestInvader / 50.0

        # 2. Total food lost to enemy
        features['lostFood'] = 1.0 - len(self.getFoodYouAreDefending(nextState).asList()) / 20.0

        # 3. Total score
        features['score'] = self.getScore(nextState) / 20.0

        # 4. Nearest food I am protecting
        #features['nearestFood'] = self.getNearestFoodDistance(nextState)

        return features


    def learn(self, state, action):
        # Get next state
        nextState = self.getSuccessor(state, action)
        agentState = nextState.getAgentState(self.index)
        features = self.getFeatures(state, action)

        # Calculate reward

        # Losing is VERY bad
        if nextState.isLost():
            print("-100 for loss")
            return -100

        # Winning is VERY good
        if nextState.isOver():
            print("100 for victory")
            return 100.0

        reward = 0.0

        # Losing food is bad
        reward -= features['lostFood'] * 5.0

        # Going near invaders is good
        reward += features['nearInvader'] * 5.0

        # Eating bad guys is good
        myFuturePos = agentState.getPosition()
        for index in self.getOpponents(state):
            a = state.getAgentState(index)
            if not a.isPacman:
                continue
            if a.getPosition() == myFuturePos:
                print("Deff: Yum, invader")
                reward += 100

        #print("Rewarded D", self.index, reward)
        self.updateWeights(state, action, nextState, reward, features)

    def final(self, gameState):
        print("\nWeights defensive")
        print(self.getWeights())
        print()
        self.gameNumber += 1
        self.alpha -= self.alphaDecay
        self.alpha = max(self.alpha, self.minAlpha)
        CaptureAgent.final(self, gameState)