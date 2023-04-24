# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # print("successorGameState: ", successorGameState)
        # print("newPos: ", newPos)
        # print("newFood: ", newFood)
        # for ghost in range(len(newGhostStates)):
        #     print("newGhostStates: ", newGhostStates[ghost].getPosition())

        # print("newScaredTimes: ", newScaredTimes)
        # # print("successorGameState.getScore(): ", successorGameState.getScore())
        # pos = currentGameState.getPacmanPosition()
        # currentScore = scoreEvaluationFunction(currentGameState)
        def distance_to_closest_food(position, food_positions):
            distances = [util.manhattanDistance(
                position, food_position) for food_position in food_positions]
            return min(distances) if distances else 0

        def get_score(position, food_positions, ghost_positions):
            score = successorGameState.getScore()
            score -= distance_to_closest_food(position, food_positions) * 3
            score += len(food_positions) * 10
            score += successorGameState.getScore() - currentGameState.getScore()
            foodList = newFood.asList()
            foodDistance = [0]
            if successorGameState.isWin():
                return 999999
            for pos in foodList:
                foodDistance.append(manhattanDistance(newPos, pos))
                ghostPos = []
            for ghost in newGhostStates:
                ghostPos.append(ghost.getPosition())

            ghostDistance = []
            for pos in ghostPos:
                ghostDistance.append(manhattanDistance(newPos, pos))

            """ Manhattan distance to each ghost in the game from current state"""
            ghostPosCurrent = []
            for ghost in currentGameState.getGhostStates():
                ghostPosCurrent.append(ghost.getPosition())
            ghostDistanceCurrent = []
            for pos in ghostPosCurrent:
                ghostDistanceCurrent.append(manhattanDistance(newPos, pos))
            numberOfFoodLeft = len(foodList)
            numberOfFoodLeftCurrent = len(currentGameState.getFood().asList())
            numberofPowerPellets = len(successorGameState.getCapsules())
            sumScaredTimes = sum(newScaredTimes)

            if sumScaredTimes > 0:
                if min(ghostDistanceCurrent) < min(ghostDistance):
                    score += 200
                else:
                    score -= 100
                # If ghosts are not scared greater distance to ghosts is better.
            else:
                if min(ghostDistanceCurrent) < min(ghostDistance):
                    score -= 100
                else:
                    score += 200

            if action == Directions.STOP:
                score -= 10
            if newPos in currentGameState.getCapsules():
                score += 400 * numberofPowerPellets
            if numberOfFoodLeft < numberOfFoodLeftCurrent:
                score += 300

            for ghost in ghost_positions:
                ghost_distance = util.manhattanDistance(
                    position, ghost.getPosition())
                if ghost_distance == 0:
                    score -= 100
                elif ghost_distance <= 2:
                    score -= 800
                elif ghost_distance <= 4:
                    score -= 150

            return score

        return get_score(newPos, newFood.asList(), newGhostStates)

        # return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    # """
    # Your minimax agent (question 2)
    # """

    # # def getAction(self, gameState: GameState):
    # """
    #     Returns the minimax action from the current gameState using self.depth
    #     and self.evaluationFunction.

    #     Here are some method calls that might be useful when implementing minimax.

    #     gameState.getLegalActions(agentIndex):
    #     Returns a list of legal actions for an agent
    #     agentIndex=0 means Pacman, ghosts are >= 1

    #     gameState.generateSuccessor(agentIndex, action):
    #     Returns the successor game state after an agent takes an action

    #     gameState.getNumAgents():
    #     Returns the total number of agents in the game

    #     gameState.isWin():
    #     Returns whether or not the game state is a winning state

    #     gameState.isLose():
    #     Returns whether or not the game state is a losing state
    #     """
    #  "*** YOUR CODE HERE ***"
    def minimax(self, agent, depth, gameState):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if agent == 0:
            actions = gameState.getLegalActions(0)
            for action in actions:
                if action == Directions.STOP:
                    actions.remove(action)

            return max(self.minimax(1, depth, gameState.generateSuccessor(agent, action)) for action in actions)
        else:
            actions = gameState.getLegalActions(0)
            for action in actions:
                if action == Directions.STOP:
                    actions.remove(action)
            nextAgent = agent + 1
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:
                depth += 5
            return min(self.minimax(nextAgent, depth, gameState.generateSuccessor(agent, action)) for action in
                       gameState.getLegalActions(agent))

        action = minimax(0, depth, gameState)
        return action

    def getAction(self, gameState):
        """
            Returns the minimax action from the current gameState using self.depth
            and self.evaluationFunction.
            """
        actions = gameState.getLegalActions(0)
        for action in actions:
            if action == Directions.STOP:
                actions.remove(action)

        scores = [self.minimax(0, 2, gameState.generateSuccessor(0, action)) for action
                  in actions]
        bestAction = max(scores)
        maxIndices = [index for index in range(
            len(scores)) if scores[index] == bestAction]
        index = random.choice(maxIndices)
        return actions[index]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning(question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        PACMAN = 0

        def max_agent(state, depth, alpha, beta):
            if state.isWin() or state.isLose():
                return state.getScore()
            actions = state.getLegalActions(PACMAN)
            best_score = float("-inf")
            score = best_score
            best_action = Directions.STOP
            for action in actions:
                score = min_agent(state.generateSuccessor(
                    PACMAN, action), depth, 1, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_action = action
                alpha = max(alpha, best_score)
                if best_score > beta:
                    return best_score
            if depth == 0:
                return best_action
            else:
                return best_score

        def min_agent(state, depth, ghost, alpha, beta):
            if state.isLose() or state.isWin():
                return state.getScore()
            next_ghost = ghost + 1
            if ghost == state.getNumAgents() - 1:
                # Although I call this variable next_ghost, at this point we are referring to a pacman agent.
                # I never changed the variable name and now I feel bad. That's why I am writing this guilty comment :(
                next_ghost = PACMAN
            actions = state.getLegalActions(ghost)
            best_score = float("inf")
            score = best_score
            for action in actions:
                if next_ghost == PACMAN:  # We are on the last ghost and it will be Pacman's turn next.
                    if depth == self.depth - 1:
                        score = self.evaluationFunction(
                            state.generateSuccessor(ghost, action))
                    else:
                        score = max_agent(state.generateSuccessor(
                            ghost, action), depth + 1, alpha, beta)
                else:
                    score = min_agent(state.generateSuccessor(
                        ghost, action), depth, next_ghost, alpha, beta)
                if score < best_score:
                    best_score = score
                beta = min(beta, best_score)
                if best_score < alpha:
                    return best_score
            return best_score
        return max_agent(gameState, 0, float("-inf"), float("inf"))


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # if currentGameState.isLose():
    #     return -float("inf")
    # elif currentGameState.isWin():
    #     return float("inf")

    # # food distance
    # foodlist = currentGameState.getFood().asList()
    # manhattanDistanceToClosestFood = min(
    #     map(lambda x: util.manhattanDistance(pos, x), foodlist))
    # distanceToClosestFood = manhattanDistanceToClosestFood

    # # number of big dots
    # # if we only count the number fo them, he'll only care about
    # # them if he has the opportunity to eat one.
    # numberOfCapsulesLeft = len(currentGameState.getCapsules())

    # # number of foods left
    # numberOfFoodsLeft = len(foodlist)

    # # ghost distance

    # # active ghosts are ghosts that aren't scared.
    # scaredGhosts, activeGhosts = [], []
    # for ghost in currentGameState.getGhostStates():
    #     if not ghost.scaredTimer:
    #         activeGhosts.append(ghost)
    #     else:
    #         scaredGhosts.append(ghost)

    # def getManhattanDistances(ghosts):
    #     return map(lambda g: util.manhattanDistance(pos, g.getPosition()), ghosts)

    # distanceToClosestActiveGhost = distanceToClosestScaredGhost = 0

    # if activeGhosts:
    #     distanceToClosestActiveGhost = min(
    #         getManhattanDistances(activeGhosts))
    # else:
    #     distanceToClosestActiveGhost = float("inf")
    # distanceToClosestActiveGhost = max(distanceToClosestActiveGhost, 5)

    # if scaredGhosts:
    #     distanceToClosestScaredGhost = min(
    #         getManhattanDistances(scaredGhosts))
    # else:
    #     # I don't want it to count if there aren't any scared ghosts
    #     distanceToClosestScaredGhost = 0

    # outputTable = [["dist to closest food", -1.5*distanceToClosestFood],
    #                ["dist to closest active ghost", 2 *
    #                    (1./distanceToClosestActiveGhost)],
    #                ["dist to closest scared ghost",
    #                    2*distanceToClosestScaredGhost],
    #                ["number of capsules left", -3.5*numberOfCapsulesLeft],
    #                ["number of total foods left", 2*(1./numberOfFoodsLeft)]]

    # score = 1 * currentScore + \
    #     -1.5 * distanceToClosestFood + \
    #     -2 * (1./distanceToClosestActiveGhost) + \
    #     -2 * distanceToClosestScaredGhost + \
    #     -20 * numberOfCapsulesLeft + \
    #     -4 * numberOfFoodsLeft
    # return score
    # util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
