# MinimumVertexCover_DRL
### Learning to tackle the Minimum Vertex Cover using Graph Convolutional Networks and RL

This repsitory contains a PyTorch implementation of an MVC environment, graph convolutional networks (using DGL) and an actor-critic algorithm. At each episode the algorithm is presented with a random Erdős-Rényi graph, with a specified number of nodes and probability of edge connection, and the neural network is trained using a simple actor-critic algorithm.
This code requires installing DGL ([Deep Graph Library](https://www.dgl.ai/)).

I have also written a [Medium article](https://towardsdatascience.com/reinforcement-learning-for-combinatorial-optimization-d1402e396e91) on the subject of reinforcement learning for combinatorial optimization, feel free to check it out.

Below are some comparisons of solutions created by the neural network policy (upper ones) and those created by a greedy heuristic (lower):

![alt text](https://user-images.githubusercontent.com/46422351/55738243-6b335600-5a2f-11e9-8c38-05aeea86c378.PNG)

---------------------------------------------------------------------------------------------------------------

![alt text](https://user-images.githubusercontent.com/46422351/55738229-61a9ee00-5a2f-11e9-93ef-2eac34fc4800.PNG)

---------------------------------------------------------------------------------------------------------------

![alt text](https://user-images.githubusercontent.com/46422351/55738172-3fb06b80-5a2f-11e9-850b-67e5dce712d0.PNG)


The yellow nodes are those chosen as part of the solution, and the darker ones were left out. During solution construction, nodes are added sequentialy until all edges are covered.
