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

import pyboltzmann as pybo

from planar_graph_sampler.combinatorial_classes.half_edge_graph import HalfEdgeGraph
from planar_graph_sampler.grammar.binary_tree_decomposition import EarlyRejectionControl
from planar_graph_sampler.grammar.grammar_utils import to_l_derived_class, divide_by_2
from planar_graph_sampler.grammar.irreducible_dissection_decomposition import irreducible_dissection_grammar
from planar_graph_sampler.operations.primal_map import PrimalMap
from planar_graph_sampler.combinatorial_classes.three_connected_graph import EdgeRootedThreeConnectedPlanarGraph


def primal_map(dissection):
    """Invokes the primal map bijection."""
    l_size = dissection.l_size
    half_edge = PrimalMap().primal_map_bijection(dissection.half_edge)
    res = EdgeRootedThreeConnectedPlanarGraph(half_edge)
    # print("{}, {}".format(l_size, res.l_size))
    # assert l_size == res.l_size
    return res


def to_bi_l_derived_class(obj):
    return to_l_derived_class(to_l_derived_class(obj))


def mark_u_atom(g_dy):
    g_dy.marked_atom = g_dy.underive_all().random_u_atom()
    return g_dy


def mark_2_u_atoms(g_dy_dy):
    he1, he2 = g_dy_dy.underive_all().two_random_u_atoms()
    assert he1 is not None and he2 is not None
    g_dy_dy.marked_atom = he1
    g_dy_dy.base_class_object.marked_atom = he2
    return g_dy_dy


def three_connected_graph_grammar():
    """Builds the three-connected planar graph grammar.

    Returns
    -------
    DecompositionGrammar
        The grammar for sampling from G_3_arrow_dx and G_3_arrow_dy
    """

    # Some shorthands to keep the grammar readable.
    Rule = pybo.AliasSampler
    J_a = Rule('J_a')
    J_a_dx = Rule('J_a_dx')
    J_a_dx_dx = Rule('J_a_dx_dx')
    G_3_arrow = Rule('G_3_arrow')
    G_3_arrow_dx = Rule('G_3_arrow_dx')
    G_3_arrow_dx_dx = Rule('G_3_arrow_dx_dx')
    G_3_arrow_dx_dy = Rule('G_3_arrow_dx_dy')
    M_3_arrow = Rule('M_3_arrow')
    M_3_arrow_dx = Rule('M_3_arrow_dx')
    M_3_arrow_dx_dx = Rule('M_3_arrow_dx_dx')
    Bij = pybo.BijectionSampler
    Rej = pybo.RejectionSampler
    Trans = pybo.TransformationSampler
    DyFromDx = pybo.UDerFromLDerSampler

    grammar = pybo.DecompositionGrammar()
    # Depends on irreducible dissection so we add those rules.
    grammar.rules = irreducible_dissection_grammar().rules
    EarlyRejectionControl.grammar = grammar

    grammar.add_rules({

        # Non-derived 3-connected rooted planar maps/graphs.

        'M_3_arrow': Bij(J_a, primal_map),

        'G_3_arrow': Trans(M_3_arrow, eval_transform=divide_by_2),  # See 4.1.9.

        # Derived 3-connected rooted planar maps/graphs.

        'M_3_arrow_dx': Bij(J_a_dx, primal_map),

        'G_3_arrow_dx': Trans(M_3_arrow_dx, to_l_derived_class, eval_transform=divide_by_2),

        'G_3_arrow_dy':
            Bij(
                DyFromDx(G_3_arrow_dx, alpha_u_l=3),  # See 5.3.3.
                mark_u_atom
            ),

        # Bi-derived 3-connected rooted planar maps/graphs.

        'M_3_arrow_dx_dx': Bij(J_a_dx_dx, primal_map),

        'G_3_arrow_dx_dx': Trans(M_3_arrow_dx_dx, to_bi_l_derived_class, eval_transform=divide_by_2),

        'G_3_arrow_dx_dy':
            Bij(
                DyFromDx(G_3_arrow_dx_dx, alpha_u_l=3),
                mark_u_atom
            ),

        'G_3_arrow_dy_dy':
            Bij(
                DyFromDx(
                    Bij(
                        G_3_arrow_dx_dy,
                        lambda gamma: gamma.invert_derivation_order()
                    ),
                    alpha_u_l=3
                ),
                mark_2_u_atoms
            ),

        # Usual 3-connected planar graphs for testing/debugging purposes.

        'G_3':
            Bij(
                Rej(
                    G_3_arrow,
                    lambda g: pybo.bern(1 / g.number_of_edges)
                ),
                lambda g: HalfEdgeGraph(g.half_edge)
            ),

    })

    return grammar


if __name__ == "__main__":
    from planar_graph_sampler.evaluations_planar_graph import *
    from timeit import default_timer as timer

    my_evals_100['G_3(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))'] = 'dummy'
    oracle = pybo.EvaluationOracle(my_evals_100)
    pybo.BoltzmannSamplerBase.oracle = oracle

    grammar = three_connected_graph_grammar()
    symbolic_x = 'x*G_1_dx(x,y)'
    symbolic_y = 'D(x*G_1_dx(x,y),y)'
    sampled_class = 'G_3_arrow'
    grammar.init(sampled_class, symbolic_x, symbolic_y)

    # random.seed(0)
    # boltzmann_framework_random_gen.seed(0)

    try:
        print("expected avg. size: {}\n".format(oracle.get_expected_l_size(sampled_class, symbolic_x, symbolic_y)))
    except pybo.PyBoltzmannError:
        pass

    l_sizes = []
    i = 0
    samples = 10000
    start = timer()
    while i < samples:
        obj = grammar.sample_iterative(sampled_class)
        # assert obj.marked_atom is not None
        l_sizes.append(obj.l_size)
        i += 1
    end = timer()
    print()
    print("avg. size: {}".format(sum(l_sizes) / len(l_sizes)))
    print("time: {}".format(end - start))

    # while True:
    #     g = grammar.sample_iterative(sampled_class, symbolic_x, symbolic_y)
    #     if g.l_size == 5:
    #         print(g)
    #         print(g.u_size / g.l_size)
    #         g = g.underive_all()
    #         # assert g.is_consistent
    #         g.plot(node_size=25, use_planar_drawer=False)
    #         plt.show()
