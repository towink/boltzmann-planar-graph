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
from framework.decomposition_grammar import AliasSampler, DecompositionGrammar

from planar_graph_sampler.grammar.binary_tree_decomposition import EarlyRejectionControl
from planar_graph_sampler.grammar.one_connected_decomposition import one_connected_graph_grammar
from planar_graph_sampler.operations.misc import relabel_networkx


def comps_to_nx_planar_embedding(components):
    """Set of connected planar graphs (possibly derived) to nx.PlanarEmbedding."""
    res = nx.PlanarEmbedding()
    for g in components:
        g = g.underive_all()
        g = g.to_planar_embedding()
        res = nx.PlanarEmbedding(nx.compose(res, g))
    relabel_networkx(res)
    return res


def comps_to_nx_graph(components):
    """Set of connected planar graphs (possibly derived) to nx.Graph."""
    res = nx.Graph()
    for g in components:
        g = g.underive_all()
        g = g.to_networkx_graph(relabel=False)
        res = nx.compose(res, g)
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
    grammar.set_builder(['G', 'G_dx', 'G_dx_dx'], PlanarGraphBuilder())


    return grammar

