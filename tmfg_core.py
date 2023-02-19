import copy
from itertools import combinations, chain

import pandas as pd
from numpy.linalg import inv

from utils import *


class TMFG:
    def __init__(self, correlation_matrix):
        self.W = correlation_matrix
        self.original_W = copy.copy(correlation_matrix)

        self.N = self.W.shape[1]
        self.P = np.zeros((self.N, self.N))
        self.max_clique_gains = np.zeros(((3 * self.N) - 6))
        self.best_vertex = np.array([-1] * ((3 * self.N) - 6))

        self.cliques = []
        self.separators = []
        self.triangles = []

        self.vertex_list = None
        self.peo = None
        self.JS = None

    def compute_TMFG(self):
        self.cliques.append(list(max_clique(self.W)))
        self.vertex_list = np.setdiff1d(range(self.N), self.cliques[0])

        self.triangles.append(list(pd.Series(self.cliques[0])[[0, 1, 2]]))
        self.triangles.append(list(pd.Series(self.cliques[0])[[0, 1, 3]]))
        self.triangles.append(list(pd.Series(self.cliques[0])[[0, 2, 3]]))
        self.triangles.append(list(pd.Series(self.cliques[0])[[1, 2, 3]]))

        self.peo = copy.copy(self.cliques[0])
        self.W = np.array(self.W)
        self.W[np.diag_indices_from(self.W)] = 0

        peo_combinations_list = []
        for n in range(len(self.cliques[0]) + 1):
            two_d_lists = len(list(combinations(self.cliques[0], n))[0])
            if two_d_lists == 2:
                peo_combinations_list += list(combinations(self.cliques[0], n))

        for i in peo_combinations_list:
            self.P[int(i[0]), int(i[1])] = self.W[int(i[0]), int(i[1])]

        for t in range(0, 4):
            index_max, max_element = get_best_gain(self.N, self.vertex_list, self.triangles[t], self.W, None)
            self.max_clique_gains[t] = max_element
            self.best_vertex[t] = index_max

        for u in range(0, (self.N - 4)):
            nt = np.argmax(self.max_clique_gains)
            nv = self.best_vertex[nt]
            self.peo.append(nv)

            thetraedron = [nv] + self.triangles[nt]
            self.cliques.append(thetraedron)
            newsep = self.triangles[nt]

            peo_combinations_list = []
            thetraedron_tbc = [nv] + newsep
            for n in range(len(thetraedron_tbc) + 1):
                two_d_lists = len(list(combinations(thetraedron_tbc, n))[0])
                if two_d_lists == 2:
                    peo_combinations_list += list(combinations(thetraedron_tbc, n))

            for i in peo_combinations_list:
                self.P[int(i[0]), int(i[1])] = self.W[int(i[0]), int(i[1])]

            self.separators.append(newsep)
            self.triangles[nt] = [newsep[0], newsep[1], nv]
            self.triangles.append([newsep[0], newsep[2], nv])
            self.triangles.append([newsep[1], newsep[2], nv])
            self.vertex_list = np.setdiff1d(self.vertex_list, nv)

            no_vertex_list = np.setdiff1d(range(self.N), self.vertex_list)

            if len(self.vertex_list) > 0:
                indices_of_interest = np.argwhere(self.best_vertex == nv)
                indices_of_interest = list(chain(*indices_of_interest))

                for t in indices_of_interest:
                    index_max, max_element = get_best_gain(self.N, self.vertex_list, self.triangles[t], self.W, no_vertex_list)
                    self.max_clique_gains[t] = max_element
                    self.best_vertex[t] = index_max

            self.max_clique_gains[nt] = 0
            ct = len(self.triangles) - 1
            if len(self.vertex_list) > 0:
                for t in [nt, (ct - 1), ct]:
                    index_max, max_element = get_best_gain(self.N, self.vertex_list, self.triangles[t], self.W, no_vertex_list)
                    self.max_clique_gains[t] = max_element
                    self.best_vertex[t] = index_max

        self.__unweighted_tmfg()
        return self.cliques, self.separators, self.JS

    def __unweighted_tmfg(self):
        self.JS = np.zeros((self.original_W.shape[0], self.original_W.shape[0]))
        for c in self.cliques:
            self.JS[np.ix_(c, c)] = 1

        np.fill_diagonal(self.JS, 0)

    def __logo(self):
        self.JS = np.zeros((self.original_W.shape[0], self.original_W.shape[0]))
        W = self.original_W.to_numpy()

        for c in self.cliques:
            self.JS[np.ix_(c, c)] = self.JS[np.ix_(c, c)] + inv(W[np.ix_(c, c)])

        for s in self.separators:
            self.JS[np.ix_(s, s)] = self.JS[np.ix_(s, s)] - inv(W[np.ix_(s, s)])

        np.fill_diagonal(self.JS, 0)