import networkx as nx
import numpy as np


def weight_my_graph(H):
    A = compute_weights(H)
    G = nx.graph(A)
    return G


def completegraph(n):
    H = np.ones((n, n))
    A = compute_weights(nx.Graph(H))
    G = nx.Graph(A)
    graph_name = "K_" + str(n) + "}"
    return G, graph_name


def cycle_graph(n):
    A = np.zeros((n, n))
    for i in range(0, n - 1):
        A[i, i + 1] = 1
    A[0, n - 1] = 1

    A = A + A.T
    G = nx.Graph(A)
    A = compute_weights(G)
    G = nx.Graph(A)
    graph_name = "C_{" + str(n) + "}"
    return G, graph_name


def gridgraph(n, m):
    G = nx.grid_2d_graph(m, n)
    G = nx.convert_node_labels_to_integers(G)
    A = compute_weights(G)
    G = nx.Graph(A)
    graph_name = "Grid_{" + str(n) + "x" + str(m) + "}"
    return G, graph_name


def compute_weights(G):
    edges = list(G.edges)
    Ei = []
    Ej = []
    for (i, j) in edges:
        Ei.append(i)
        Ej.append(j)

    WA = np.zeros((G.number_of_nodes(), G.number_of_nodes()))

    for e in range(G.number_of_edges()):
        WA[Ei[e], Ej[e]] = 1 / (1 + max(G.degree(Ei[e]), G.degree(Ej[e])))

    WA = WA + WA.T

    for i in range(G.number_of_nodes()):
        neighbors_i = [j for j in G.neighbors(i)]
        WA[i, i] = 1 - sum([WA[i, j] for j in neighbors_i])

    return WA
