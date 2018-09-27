# -*- coding: utf-8 -*-
#    Copyright (C) 2018 by
#    Marta Grobelna <marta.grobelna@rwth-aachen.de>
#    Petre Petrov <petrepp4@gmail.com>
#    Rudi Floren <rudi.floren@gmail.com>
#    Tobias Winkler <tobias.winkler1@rwth-aachen.de>
#    All rights reserved.
#    BSD license.
#
# Authors:  Marta Grobelna <marta.grobelna@rwth-aachen.de>
#           Petre Petrov <petrepp4@gmail.com>
#           Rudi Floren <rudi.floren@gmail.com>
#           Tobias Winkler <tobias.winkler1@rwth-aachen.de>

import math
import multiprocessing as multiproc

from planar_graph_sampler.grammar.planar_graph_decomposition import \
    planar_graph_grammar, comps_to_nx_planar_embedding, comps_to_nx_graph
from pyboltzmann.evaluation_oracle import EvaluationOracle
from pyboltzmann.generic_samplers import BoltzmannSamplerBase
from pyboltzmann.generic_classes import SetClass
from planar_graph_sampler.evaluations_planar_graph import *
import networkx as nx
import datetime

__all__ = ['random_planar_graph',
           'PlanarGraphGenerator']


def random_planar_graph(n, epsilon=0.1, require_connected=True,
                        with_embedding=True, allow_multiproc=False):
    """See PlanarGraphGenerator."""
    return PlanarGraphGenerator(n, epsilon, require_connected,
                                with_embedding, allow_multiproc).sample()


class PlanarGraphGenerator(object):
    """
    Sets up a uniform random generator for planar graphs.

    Parameters
    ----------
    n : int
        Number of nodes, must be at least 3
    epsilon : float, optional (default=0.1)
        Enables approximate size sampling of graphs with number of nodes in the
        interval [n(1-eps), n(1+eps)]. Set to a value smaller than 1/n for
        exact size sampling.
    require_connected : bool, optional (default=False)
        Sample from the class of connected planar graphs instead of general
        planar graphs.
    with_embedding : bool, optional (default=True)
        If set to True, the generated graphs are returned with an arbitrary
        planar embedding represented as an nx.PlanarEmbedding object.
        Otherwise an nx.Graph object is returned.
    allow_multiproc : bool, optional (default=False)
        Allows usage of the multiprocessing module for parallel sampling.

    Returns
    -------
    G : Graph
        Planar graph drawn uniformly at random. The nodes are labeled with
        consecutive integers starting from 1. If `with_embedding` is set to
        `True`, an ``nx.PlanarEmbedding`` object is returned, otherwise
        ``nx.Graph``.

    Notes
    -----
    Expected running time is O(n/eps).
    In particular, the expected time for exact size sampling is O(n²).

    References
    ----------
    .. [1] E. Fusy:
        Uniform random sampling of planar graphs in linear time
    """

    def __init__(self, n, epsilon=0.1, require_connected=True,
                 with_embedding=True, allow_multiproc=False):
        # Handle invalid arguments.
        if n < 3:
            raise ValueError("n must be an integer greater or equal than 3")
        self._n = n
        if epsilon > 1 or epsilon < 0:
            raise ValueError("epsilon must be a real number in [0,1]")
        self._eps = epsilon

        # Compute interval of accepted sizes.
        self._lower = math.ceil(n * (1 - epsilon))
        self._upper = math.floor(n * (1 + epsilon))

        self._require_connected = require_connected
        self._with_embedding = with_embedding
        self._allow_parallel = allow_multiproc
        if allow_multiproc:
            self.sample = self._sample_multiproc
        else:
            self.sample = self._sample_single_proc

        # Set up the oracle and grammar for sampling.
        # BoltzmannSamplerBase.oracle = \
        #    EvaluationOracle.get_best_oracle_for_size(n, planar_graph_evals)
        # TODO Implement the choice of values.
        BoltzmannSamplerBase.oracle = EvaluationOracle(my_evals_1000)
        self._grammar = planar_graph_grammar()
        self._grammar.init()
        self._grammar._precompute_evals('G_dx_dx_dx', 'x', 'y')

    def sample(self):
        """Invokes the random generator once."""
        # This method is set in __init__.
        pass

    def _sample_single_proc(self):
        if self._require_connected:
            while True:
                half_edge_graph = self._grammar.sample_iterative(
                    'G_1_dx_dx_dx').underive_all()
                if self._lower <= half_edge_graph.number_of_nodes <= self._upper:
                    if self._with_embedding:
                        return half_edge_graph.to_planar_embedding(relabel=False)
                    else:
                        return half_edge_graph.to_networkx_graph(relabel=True)
        else:
            while True:
                generic_set = self._grammar.sample_iterative(
                    'G_dx_dx_dx')
                if self._lower <= generic_set.l_size + 3 <= self._upper:
                    if self._with_embedding:
                        return comps_to_nx_planar_embedding(generic_set)
                    else:
                        return comps_to_nx_graph(generic_set)

    def _sample_multiproc(self):
        cpu_count = multiproc.cpu_count()
        processes_queue = multiproc.Queue()

        def multiproc_target_func(self, queue):
            res = self._sample_single_proc()
            queue.put(res)

        processes = [
            multiproc.Process(
                target=multiproc_target_func,
                args=(self, processes_queue))
            for _ in range(cpu_count)]

        for p in processes:
            p.daemon = True
            p.start()

        # Wait for the first result.
        result = processes_queue.get()
        # Terminate the other processes and return result.
        for p in processes:
            p.terminate()
        return result


if __name__ == "__main__":
    generator = PlanarGraphGenerator(
        1000, require_connected=True, with_embedding=False)

    while True:
        G = generator.sample()
        print(G.number_of_edges() / G.number_of_nodes())

        assert nx.check_planarity(G)

        # import matplotlib.pyplot as plt
        # # pos = nx.combinatorial_embedding_to_pos(G, fully_triangulate=False)
        # nx.draw(G, with_labels=True)
        # plt.show()

