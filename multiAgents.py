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
        # our main score variable
        score = successorGameState.getScore()

        # Get food and capsules positions, distances, min distances, and the amount left
        foodList = newFood.asList()
        foodDistances = [util.manhattanDistance(
            newPos, food) for food in foodList]
        minFoodDistance = min(foodDistances) if len(foodDistances) > 0 else 0
        numFoodLeft = len(foodList)
        capsuleList = currentGameState.getCapsules()
        numCapusles = len(capsuleList)
        numCapsulesLeft = len(successorGameState.getCapsules())

        # Get ghost positions and distances, min distances, and scared ghost distances
        scaredGhosts = [
            ghost for ghost in newGhostStates if ghost.scaredTimer > 0]
        ghostPositions = [ghost.getPosition() for ghost in newGhostStates]
        ghostDistances = [util.manhattanDistance(
            newPos, ghost) for ghost in ghostPositions]
        minGhostDistance = min(ghostDistances) if len(
            ghostDistances) > 0 else 0
        scaredGhostDistances = [util.manhattanDistance(
            newPos, ghost.getPosition()) for ghost in scaredGhosts]
        minScaredGhostDistance = min(scaredGhostDistances) if len(
            scaredGhostDistances) > 0 else 0

        # Update score based on various conditions (I tried different numbers by trial and error)
        score -= minFoodDistance * 5
        # stay away from ghosts
        if minGhostDistance == 0:
            score -= 10000
        elif minGhostDistance <= 2:
            score -= 800
        elif minGhostDistance <= 4:
            score -= 200

        if minFoodDistance <= 3:
            score += 10*(1/minFoodDistance)
    # if the ghost is scared and there is enough time, eat it.
        if min(newScaredTimes) > 1:
            # eat scared ghosts
            if minGhostDistance < 3:
                score += 1500

        if action == Directions.STOP:
            # penalize pacman for wasting time
            score -= 10
        if numFoodLeft < currentGameState.getNumFood():
            # encourage pacman to eat food
            score += 300
        if numCapsulesLeft < numCapusles:
            # encourage pacman to eat power capsules
            score += 1000
        else:
            # penalize pacman for wasting time
            score -= 100

        return score


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
            return max(self.minimax(1, depth, gameState.generateSuccessor(agent, action)) for action in actions)
        else:
            actions = gameState.getLegalActions(0)
            nextAgent = agent + 1
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            depth += 1
            return min(self.minimax(nextAgent, depth, gameState.generateSuccessor(agent, action)) for action in
                       gameState.getLegalActions(agent))

    def getAction(self, gameState):
        """
            Returns the minimax action from the current gameState using self.depth
            and self.evaluationFunction.
            """
        actions = gameState.getLegalActions(0)
        scores = [self.minimax(0, 0, gameState.generateSuccessor(0, action)) for action
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

    def alphaBeta(self, agent, depth, gameState, alpha, beta):

        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(agent)

        if agent == 0:
            score = max(self.alphaBeta(1, depth, gameState.generateSuccessor(
                agent, action), alpha, beta) for action in actions)
            alpha = max(alpha, score)
            if beta < alpha:  # pruning
                return score
            if score > beta:
                return score
            return score
        else:
            nextAgent = agent + 1
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            depth += 1
            score = min(self.alphaBeta(nextAgent, depth, gameState.generateSuccessor(agent, action), alpha, beta) for action in
                        gameState.getLegalActions(agent))
            beta = min(beta, score)
            if beta < alpha:  # pruning
                return score
            if score < alpha:
                return score
            return score

    def getAction(self, gameState):
        """
            Returns the minimax action from the current gameState using self.depth
            and self.evaluationFunction.
        """
        actions = gameState.getLegalActions(0)
        scores = [self.alphaBeta(0, 0, gameState.generateSuccessor(
            0, action), -999999, 9999999) for action in actions]
        bestAction = max(scores)
        maxIndices = [index for index in range(
            len(scores)) if scores[index] == bestAction]
        index = random.choice(maxIndices)
        return actions[index]


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

    def expectimax(self, agent, depth, gameState):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        if agent == 0:
            legalActions = gameState.getLegalActions(0)
            legalActions.remove(Directions.STOP)
            maxScore = -999999

            for action in legalActions:
                successor = gameState.generateSuccessor(0, action)
                score = self.expectimax(1, depth, successor)
                maxScore = max(maxScore, score)

            return maxScore

        else:
            successorStates = gameState.getLegalActions(agent)
            expectedScore = 0

            for state in successorStates:
                successor = gameState.generateSuccessor(agent, state)

                if agent == gameState.getNumAgents() - 1:
                    score = self.expectimax(0, depth + 1, successor)
                else:
                    score = self.expectimax(agent + 1, depth, successor)

                expectedScore += score

        return expectedScore / len(successorStates)

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        legalActions = gameState.getLegalActions(0)
        legalActions.remove(Directions.STOP)
        depth = gameState.getNumAgents() * self.depth
        maxScore = -999999
        returnAction = ''

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            score = self.expectimax(1, 0, successor)

            if score > maxScore:
                maxScore = score
                returnAction = action

        return returnAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    The weights assigned to each factor are empirically chosen and can be tweaked to adjust the behavior of the agent. 
    The closer the distance to food and farther the distance to active ghost, the higher the score will be.
    Additionally, if there are more capsules left, the score will decrease,
    while if there are more food pellets left, the score will increase.


    """
    "*** YOUR CODE HERE ***"
    pacman_pos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()

    if currentGameState.isLose():
        return -float('inf')
    elif currentGameState.isWin():
        return float('inf')

    food_list = currentGameState.getFood().asList()
    distance_to_closest_food = min(
        [util.manhattanDistance(pacman_pos, food) for food in food_list])

    ghostsList = []
    scaredGhosts = []

    for ghost in currentGameState.getGhostStates():
        if not ghost.scaredTimer:
            ghostsList.append(ghost)
        else:
            scaredGhosts.append(ghost)

    distance_to_closest_active_ghost = float('inf')
    if ghostsList:
        distance_to_closest_active_ghost = min([util.manhattanDistance(
            pacman_pos, ghost.getPosition()) for ghost in ghostsList])

    distance_to_closest_scared_ghost = 0
    if scaredGhosts:
        distance_to_closest_scared_ghost = min([util.manhattanDistance(
            pacman_pos, ghost.getPosition()) for ghost in scaredGhosts])

    number_of_capsules_left = len(currentGameState.getCapsules())
    number_of_foods_left = len(food_list)

    # Empirically chosen weights for the factors
    active_ghost_weight = -2.5
    scared_ghost_weight = -2.5
    food_weight = -1.5
    capsule_weight = -15
    food_left_weight = -4

    score += active_ghost_weight / max(distance_to_closest_active_ghost, 1)
    score += scared_ghost_weight * distance_to_closest_scared_ghost
    score += food_weight * distance_to_closest_food
    score += capsule_weight * number_of_capsules_left
    score += food_left_weight * number_of_foods_left

    return score


# Abbreviation
better = betterEvaluationFunction
