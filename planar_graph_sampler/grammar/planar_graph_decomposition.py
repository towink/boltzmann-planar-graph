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

import networkx as nx

from framework.generic_samplers import *
from framework.decomposition_grammar import DecompositionGrammar
from framework.generic_samplers import AliasSampler

from planar_graph_sampler.grammar.binary_tree_decomposition import EarlyRejectionControl
from planar_graph_sampler.grammar.one_connected_decomposition import one_connected_graph_grammar
from planar_graph_sampler.operations.misc import relabel_networkx


def comps_to_nx_planar_embedding(components):
    """Set of connected planar graphs (possibly derived) to nx.PlanarEmbedding."""
    res = nx.PlanarEmbedding()
    for g in components:
        g = g.underive_all()
        g = g.to_planar_embedding(relabel=False)
        res = nx.PlanarEmbedding(nx.compose(res, g))
    # Relabel once all the components are in the same graph.
    # relabel_networkx(res)
    return res


def comps_to_nx_graph(components):
    """Set of connected planar graphs (possibly derived) to nx.Graph."""
    res = nx.Graph()
    for g in components:
        g = g.underive_all()
        g = g.to_networkx_graph(relabel=False)
        res = nx.compose(res, g)
    # Relabel once all the components are in the same graph.
    relabel_networkx(res)
    return res


class PlanarGraphBuilder(DefaultBuilder):
    def product(self, lhs, rhs):
        # Treat products like sets.
        rhs.append(lhs)
        return rhs


def planar_graph_grammar():
    """Constructs the grammar for planar graphs.

    Returns
    -------
    grammar : DecompositionGrammar
        The grammar for sampling from G, G_dx and G_dx_dx.
    """

    # Some shortcuts to make the grammar more readable.
    G_1 = AliasSampler('G_1')
    G_1_dx = AliasSampler('G_1_dx')
    G_1_dx_dx = AliasSampler('G_1_dx_dx')
    G_1_dx_dx_dx = AliasSampler('G_1_dx_dx_dx')
    G = AliasSampler('G')
    G_dx = AliasSampler('G_dx')
    G_dx_dx = AliasSampler('G_dx')

    grammar = DecompositionGrammar()
    grammar.rules = one_connected_graph_grammar().rules
    EarlyRejectionControl.grammar = grammar

    grammar.rules = {

        'G': SetSampler(0, G_1),

        'G_dx': G_1_dx * G,

        'G_dx_dx': G_1_dx_dx * G + G_1_dx * G_dx,

        'G_dx_dx_dx': G_1_dx_dx_dx * G + G_1_dx_dx * G_dx + G_1_dx_dx * G_dx + G_1_dx * G_dx_dx

    }
    grammar.set_builder(
        ['G', 'G_dx', 'G_dx_dx', 'G_dx_dx_dx'], PlanarGraphBuilder())

    return grammar


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from planar_graph_sampler.evaluations_planar_graph import *
    from timeit import default_timer as timer
    from framework.evaluation_oracle import EvaluationOracle

    oracle = EvaluationOracle(my_evals_100)
    BoltzmannSamplerBase.oracle = oracle
    BoltzmannSamplerBase.debug_mode = False

    start = timer()
    grammar = planar_graph_grammar()
    grammar.init()
    symbolic_x = 'x'
    symbolic_y = 'y'
    sampled_class = 'G_dx_dx_dx'
    # symbolic_x = 'x*G_1_dx(x,y)'
    # symbolic_y = 'D(x*G_1_dx(x,y),y)'
    # sampled_class = 'K'
    # print(grammar.collect_oracle_queries(sampled_class, symbolic_x, symbolic_y))
    grammar.precompute_evals(sampled_class, symbolic_x, symbolic_y)
    end = timer()
    print("Time init: {}".format(end - start))

    try:
        print("expected avg. size: {}\n".format(oracle.get_expected_l_size(sampled_class, symbolic_x, symbolic_y)))
    except BoltzmannFrameworkError:
        pass

    # random.seed(0)
    # boltzmann_framework_random_gen.seed(13)

    l_sizes = []
    i = 0
    samples = 10000000
    start = timer()
    while i < samples:
        obj = grammar.sample_iterative(sampled_class, symbolic_x, symbolic_y)
        l_sizes.append(obj.l_size)
        if obj.l_size > 100:
            print(obj.l_size)
        i += 1
        # if obj.l_size == 0:
        #     G = comps_to_nx_graph(obj)
        #     nx.draw(G)
        #     plt.show()
    end = timer()
    print()
    print("avg. size: {}".format(sum(l_sizes) / len(l_sizes)))
    print("time: {}".format(end - start))
