# MinimumVertexCover_DRL
### Learning to tackle the Minimum Vertex Cover using Graph Convolutional Networks and RL

This repsitory contains a PyTorch implementation of an MVC environment, graph convolutional networks (using DGL) and an actor-critic algorithm. At each episode the algorithm is presented with a random Erdős-Rényi graph, with a specified number of nodes and probability of edge connection, and the neural network is trained using a simple actor-critic algorithm.
This code requires installing DGL ([Deep Graph Library](https://www.dgl.ai/)).
