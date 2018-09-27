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

from __future__ import division

import pyboltzmann as pybo

from planar_graph_sampler.grammar.binary_tree_decomposition import EarlyRejectionControl
from planar_graph_sampler.grammar.grammar_utils import Counter, divide_by_2, to_u_derived_class
from planar_graph_sampler.combinatorial_classes.halfedge import HalfEdge
from planar_graph_sampler.combinatorial_classes.two_connected_graph import EdgeRootedTwoConnectedPlanarGraph, \
    TwoConnectedPlanarGraph
from planar_graph_sampler.grammar.network_decomposition import network_grammar


class ZeroAtomGraphBuilder(pybo.DefaultBuilder):
    """Builds zero atoms of the class G_2_arrow (= link graphs)."""

    def __init__(self):
        self._counter = Counter()

    def zero_atom(self):
        # TODO is this really the correct zero atom?
        root_half_edge = HalfEdge()
        root_half_edge.next = root_half_edge
        root_half_edge.prior = root_half_edge
        root_half_edge.node_nr = next(self._counter)
        root_half_edge_opposite = HalfEdge()
        root_half_edge_opposite.next = root_half_edge_opposite
        root_half_edge_opposite.prior = root_half_edge_opposite
        root_half_edge_opposite.node_nr = next(self._counter)
        root_half_edge.opposite = root_half_edge_opposite
        root_half_edge_opposite.opposite = root_half_edge
        return EdgeRootedTwoConnectedPlanarGraph(root_half_edge)


def to_G_2(decomp):
    return TwoConnectedPlanarGraph(decomp.second.half_edge)


def to_G_2_dx(decomp):
    g = decomp.second.underive_all()
    assert isinstance(g, EdgeRootedTwoConnectedPlanarGraph), g
    return pybo.LDerivedClass(TwoConnectedPlanarGraph(g.half_edge))


def to_G_2_dx_dx(decomp):
    if isinstance(decomp, pybo.ProdClass):
        g = decomp.second
    else:
        g = decomp
    g = g.underive_all()
    assert isinstance(g, EdgeRootedTwoConnectedPlanarGraph), g
    return pybo.LDerivedClass(pybo.LDerivedClass(TwoConnectedPlanarGraph(g.half_edge)))


def to_G_2_arrow(network):
    # TODO isn't network already EdgeRootedTwoConnectedPlanarGraph when sampling from Z?
    return EdgeRootedTwoConnectedPlanarGraph(network.half_edge)


def to_G_2_arrow_dx(network):
    return pybo.LDerivedClass(to_G_2_arrow(network))


def to_G_2_arrow_dx_dx(network):
    return pybo.LDerivedClass(to_G_2_arrow_dx(network))


def divide_by_1_plus_y(evl, x, y):
    """Needed as an eval-transform for rules G_2_arrow and G_2_arrow_dx."""
    return evl / (1 + pybo.BoltzmannSamplerBase.oracle.get(y))


def mark_l_atom(g_dx):
    g_dx.marked_atom = g_dx.underive_all().random_node_half_edge(1)
    return g_dx


def mark_2_l_atoms(g_dx_dx):
    atoms = g_dx_dx.underive_all().random_node_half_edge(2)
    g_dx_dx.marked_atom = atoms[0]
    g_dx_dx.base_class_object.marked_atom = atoms[1]
    return g_dx_dx


def mark_3_l_atoms(g_dx_dx_dx):
    atoms = g_dx_dx_dx.underive_all().random_node_half_edge(3)
    g_dx_dx_dx.marked_atom = atoms[0]
    g_dx_dx_dx.base_class_object.marked_atom = atoms[1]
    g_dx_dx_dx.base_class_object.base_class_object.marked_atom = atoms[2]
    return g_dx_dx_dx


def two_connected_graph_grammar():
    """Constructs the grammar for two connected planar graphs.

    Returns
    -------
    DecompositionGrammar
        The grammar for sampling from G_2_dx and G_2_dx_dx.
    """

    Z = pybo.ZeroAtomSampler
    L = pybo.LAtomSampler
    Rule = pybo.AliasSampler
    D = Rule('D')
    D_dx = Rule('D_dx')
    D_dx_dx = Rule('D_dx_dx')
    F = Rule('F')
    F_dx = Rule('F_dx')
    F_dx_dx = Rule('F_dx_dx')
    G_2_dy = Rule('G_2_dy')
    G_2_dx_dy = Rule('G_2_dx_dy')
    G_2_dx_dx_dy = Rule('G_2_dx_dx_dy')
    G_2_arrow = Rule('G_2_arrow')
    G_2_arrow_dx = Rule('G_2_arrow_dx')
    G_2_arrow_dx_dx = Rule('G_2_arrow_dx_dx')
    Trans = pybo.TransformationSampler
    Bij = pybo.BijectionSampler
    DxFromDy = pybo.LDerFromUDerSampler

    grammar = pybo.DecompositionGrammar()
    grammar.rules = network_grammar().rules
    EarlyRejectionControl.grammar = grammar

    grammar.add_rules({

        # two connected

        'G_2_arrow': Trans(Z() + D, to_G_2_arrow, eval_transform=divide_by_1_plus_y),  # see 5.5

        'F': Bij(L() ** 2 * G_2_arrow, to_G_2),

        'G_2_dy': Trans(F, to_u_derived_class, eval_transform=divide_by_2),

        'G_2_dx':
            Bij(
                DxFromDy(G_2_dy, alpha_l_u=2.0),  # see p. 26 TODO check this, error in paper? (no error, 2 is caused by the link graph)
                mark_l_atom
            ),

        # l-derived two connected

        'G_2_arrow_dx': Trans(D_dx, to_G_2_arrow_dx, eval_transform=divide_by_1_plus_y),

        'F_dx': Bij(L() ** 2 * G_2_arrow_dx + 2 * L() * G_2_arrow, to_G_2_dx),

        'G_2_dx_dy': Trans(F_dx, to_u_derived_class, eval_transform=divide_by_2),

        'G_2_dx_dx':
            Bij(
                DxFromDy(G_2_dx_dy, alpha_l_u=1.0),  # see 5.5
                mark_2_l_atoms
            ),

        # bi-l-derived two connected

        'G_2_arrow_dx_dx': Trans(D_dx_dx, to_G_2_arrow_dx_dx, eval_transform=divide_by_1_plus_y),

        'F_dx_dx': Bij(L() ** 2 * G_2_arrow_dx_dx + 4 * L() * G_2_arrow_dx + 2 * G_2_arrow, to_G_2_dx_dx),

        'G_2_dx_dx_dy': Trans(F_dx_dx, to_u_derived_class, eval_transform=divide_by_2),

        'G_2_dx_dx_dx':
            Bij(
                DxFromDy(G_2_dx_dx_dy, alpha_l_u=1.0),
                mark_3_l_atoms
            ),

    })

    grammar.set_builder(['G_2_arrow'], ZeroAtomGraphBuilder())


    return grammar


if __name__ == '__main__':
    from planar_graph_sampler.evaluations_planar_graph import *
    from timeit import default_timer as timer

    oracle = pybo.EvaluationOracle(my_evals_100)
    pybo.BoltzmannSamplerBase.oracle = oracle
    pybo.BoltzmannSamplerBase.debug_mode = False

    start = timer()
    grammar = two_connected_graph_grammar()
    symbolic_x = 'x*G_1_dx(x,y)'
    symbolic_y = 'y'
    sampled_class = 'G_2_dx'
    grammar.init(sampled_class, symbolic_x, symbolic_y)
    end = timer()
    print("Time init: {}".format(end - start))

    try:
        print("expected avg. size: {}\n".format(oracle.get_expected_l_size(sampled_class, symbolic_x, symbolic_y)))
    except pybo.PyBoltzmannError:
        pass

    # random.seed(0)
    # boltzmann_framework_random_gen.seed(13)

    l_sizes = []
    i = 0
    samples = 100
    start = timer()
    while i < samples:
        obj = grammar.sample_iterative(sampled_class)
        l_sizes.append(obj.l_size)
        # print(obj.l_size)
        i += 1
    end = timer()
    print()
    print("avg. size: {}".format(sum(l_sizes) / len(l_sizes)))
    print("time: {}".format(end - start))

    # while True:
    #     g = grammar.sample_iterative(sampled_class, symbolic_x, symbolic_y)
    #     g = g.underive_all()
    #     if g.l_size == 4:
    #         print(g)
    #         # assert g.is_consistent
    #         g.plot(with_labels=True, use_planar_drawer=False, node_size=25)
    #         plt.show()
