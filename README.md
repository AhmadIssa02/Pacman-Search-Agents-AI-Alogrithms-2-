NOTE: My agent is ExpectimaxAgent using the better evaluation function
TO RUN MY AGENT, USE THE FOLLOWING COMMAND python pacman.py -p ExpectimaxAgent -l mediumClassic -a evalFn=better

########Question 1:
The function first initializes the score variable as the current game state's score. It then extracts various pieces of information from the game state, including the positions of food, capsules, and ghosts, as well as the distances to those objects.

The score is then updated based on various conditions. For example, the score is decreased by a multiple of the minimum distance to the nearest food item, to incentivize Pacman to move towards food. Similarly, if the minimum distance to a ghost is zero (meaning Pacman is on the same spot as a ghost), the score is heavily penalized. If the distance to a scared ghost is less than three, the score is increased to encourage Pacman to eat the ghost while it is vulnerable.

Other factors that influence the score include the amount of food and capsules remaining, and penalties for wasting time by stopping or not eating.

The scoring function is designed to balance the competing objectives of eating food, avoiding ghosts, and taking advantage of opportunities to eat vulnerable ghosts. The function was tuned through trial and error to achieve a balance between these objectives that produced effective AI gameplay.

########Question 2:
This code implements the minimax algorithm to determine the best action for a player in a two-player game with alternating turns. The function minimax recursively computes the best score that a player can achieve given a certain game state, agent, and depth. The function returns the evaluation score if the game is won, lost, or the depth is reached. Otherwise, the function checks the legal actions of the agent, and if the agent is the maximizing player, it selects the action that maximizes the score, otherwise, it selects the action that minimizes the score, and it continues to the next agent.

The getAction function generates the successor game states for each possible action of the player, and then it computes the scores of each of these successor states using the minimax function. It selects the best score and returns the corresponding action. If there are multiple actions with the same best score, it chooses one at random.

########Question 3:

The function first checks if the game is over or if the maximum search depth has been reached. If either of these conditions is true, the function returns the evaluation of the current game state using the evaluationFunction method.

If the game is not over and the maximum depth has not been reached, the function generates the next possible game states and calculates the score for each state using the alphaBeta function. If the current agent is the maximizing agent (agent 0), the function returns the maximum score of the possible game states. It also updates the alpha value and prunes irrelevant branches of the game tree if necessary.

If the current agent is the minimizing agent, the function returns the minimum score of the possible game states. It also updates the beta value and prunes irrelevant branches of the game tree if necessary.

Finally, the getAction method uses the alphaBeta function to calculate the score for each possible action of agent 0 in the current game state. It then chooses a random action from the actions that have the highest score and returns that action.

########Question 4:
The function first checks if the game is over or if the maximum search depth has been reached. If either of these conditions is true, the function returns the evaluation of the current game state using the evaluationFunction method.

If the game is not over and the maximum depth has not been reached, the function generates the next possible game states and calculates the score for each state using the expectimax function. If the current agent is the maximizing agent (agent 0), the function returns the maximum score of the possible game states.

If the current agent is a ghost, the function calculates the expected score for the possible game states using the average of the scores returned by the expectimax function for the other ghosts. The function returns the expected score.

Finally, the getAction method uses the expectimax function to calculate the score for each possible action of agent 0 in the current game state. It then chooses the action with the highest score and returns that action.

Note: This implementation models all ghosts as choosing uniformly at random from their legal moves.

########Question 5:
The score is determined by considering various factors such as the distance to the closest food, the distance to the closest active ghost, the distance to the closest scared ghost, the number of capsules left, and the number of food pellets left.

If the current game state is a winning state, the function returns positive infinity, and if it's a losing state, it returns negative infinity.

The weights assigned to each factor are empirically chosen and can be tweaked to adjust the behavior of the agent. The closer the distance to food and farther the distance to active ghost, the higher the score will be. Additionally, if there are more capsules left, the score will decrease, while if there are more food pellets left, the score will increase.
