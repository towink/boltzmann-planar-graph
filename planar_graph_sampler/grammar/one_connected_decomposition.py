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

from planar_graph_sampler.combinatorial_classes.one_connected_graph import OneConnectedPlanarGraph
from planar_graph_sampler.grammar.binary_tree_decomposition import EarlyRejectionControl
from planar_graph_sampler.grammar.grammar_utils import underive
from planar_graph_sampler.grammar.two_connected_decomposition import two_connected_graph_grammar


class Merger(pybo.DefaultBuilder):
    """
    Merges a set of l-derived graphs at their marked vertices.
    """
    def set(self, graphs):
        # Merge a set of l-derived one-connected planar graphs at their marked vertices.
        # If the set is empty, return a single-node graph.
        if len(graphs) is 0:
            g = OneConnectedPlanarGraph()
            return pybo.LDerivedClass(OneConnectedPlanarGraph(), g.half_edge)
        result = graphs.pop()
        for g in graphs:
            result.marked_atom.insert_all_after(g.marked_atom)
        assert isinstance(result, pybo.LDerivedClass)
        return result


def merge(prod):
    """Merges l-derived one-connected graphs at their marked vertices"""
    # lhs is a bi-derived connected and rhs a derived connected.
    lhs = prod.first
    rhs = prod.second
    if not lhs.marked_atom.is_trivial:
        rhs.marked_atom.insert_all_after(lhs.marked_atom)
    return lhs


def subs_marked_vertex(decomp):
    # decomp is of form (G_1_dx + L * G_1_dx_dx) * G_2_dx_dx.

    if isinstance(decomp.first, pybo.LDerivedClass):
        one_connected = decomp.first
        plug_in_he = one_connected.marked_atom
        if not plug_in_he.is_trivial:
            decomp.second.base_class_object.marked_atom.insert_all_after(plug_in_he)
    else:
        one_connected = decomp.first.second
        plug_in_he = one_connected.marked_atom
        if not plug_in_he.is_trivial:
            decomp.second.base_class_object.marked_atom.insert_all_after(plug_in_he)
            decomp.second.base_class_object.marked_atom = one_connected.base_class_object.marked_atom
    return decomp.second


def subs_marked_vertex_2(decomp):
    # decomp is of form ((G_1_dx + L * G_1_dx_dx) * (G_1_dx + L * G_1_dx_dx)) * G_2_dx_dx_dx.
    if isinstance(decomp.first.first, pybo.LDerivedClass):
        plug_in_he1 = decomp.first.first.marked_atom
    else:
        plug_in_he1 = decomp.first.first.second.marked_atom
    if isinstance(decomp.first.second, pybo.LDerivedClass):
        plug_in_he2 = decomp.first.second.marked_atom
    else:
        plug_in_he2 = decomp.first.second.second.marked_atom
    if not plug_in_he1.is_trivial:
        decomp.second.base_class_object.base_class_object.marked_atom.insert_all_after(plug_in_he1)
    if not plug_in_he2.is_trivial:
        decomp.second.base_class_object.marked_atom.insert_all_after(plug_in_he2)
    return decomp.second


def rej_to_G_1(g):
    return pybo.bern(1 / (g.l_size + 1))


def one_connected_graph_grammar():
    """Constructs the grammar for connected planar graphs.

    Returns
    -------
    DecompositionGrammar
        The grammar for sampling from G_1_dx and G_1_dx_dx.
    """

    # Some shortcuts to make the grammar more readable.
    L = pybo.LAtomSampler
    Rule = pybo.AliasSampler
    G_2_dx = Rule('G_2_dx')
    G_2_dx_dx = Rule('G_2_dx_dx')
    G_2_dx_dx_dx = Rule('G_2_dx_dx_dx')
    G_1_dx = Rule('G_1_dx')
    G_1_dx_dx = Rule('G_1_dx_dx')
    G_1_dx_dx_dx = Rule('G_1_dx_dx_dx')
    Set = pybo.SetSampler
    LSubs = pybo.LSubsSampler
    Bij = pybo.BijectionSampler
    Rej = pybo.RejectionSampler

    grammar = pybo.DecompositionGrammar()
    grammar.rules = two_connected_graph_grammar().rules
    EarlyRejectionControl.grammar = grammar

    grammar.add_rules({

        'G_1':
            Bij(
                Rej(
                    G_1_dx,
                    rej_to_G_1  # See lemma 15.
                ),
                underive
            ),

        'G_1_dx':
            Set(
                0,
                LSubs(
                    G_2_dx,
                    L() * G_1_dx
                )
            ),

        'G_1_dx_dx':
            Bij(
                Bij(
                    (G_1_dx + L() * G_1_dx_dx) * LSubs(G_2_dx_dx, L() * G_1_dx),
                    subs_marked_vertex
                ) * G_1_dx,
                merge
            ),

        'G_1_dx_dx_dx':
            Bij(
                Bij(
                    (2 * G_1_dx_dx + L() * G_1_dx_dx_dx) * LSubs(G_2_dx_dx, L() * G_1_dx),
                    subs_marked_vertex
                ) * G_1_dx,
                merge
            )

            + Bij(
                Bij(
                    (G_1_dx + L() * G_1_dx_dx) ** 2 * LSubs(G_2_dx_dx_dx, L() * G_1_dx),
                    subs_marked_vertex_2
                ) * G_1_dx,
                merge
            )

            + Bij(
                Bij(
                    (G_1_dx + L() * G_1_dx_dx) * LSubs(G_2_dx_dx, L() * G_1_dx),
                    subs_marked_vertex
                ) * G_1_dx_dx,
                merge
            ),

    })

    grammar.set_builder(['G_1_dx'], Merger())

    return grammar


if __name__ == '__main__':
    from planar_graph_sampler.evaluations_planar_graph import *
    from timeit import default_timer as timer

    oracle = pybo.EvaluationOracle(my_evals_10)
    pybo.BoltzmannSamplerBase.oracle = oracle
    pybo.BoltzmannSamplerBase.debug_mode = False

    start = timer()
    grammar = one_connected_graph_grammar()
    symbolic_x = 'x'
    symbolic_y = 'y'
    sampled_class = 'G_1_dx_dx_dx'
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
    #     print(g)
    #     for key, value in sorted(Stats.rules.items(), key=lambda x: x[1]):
    #         print("{} : {}".format(key, value))
    #     print()
    #     # if g.l_size >= 1000:
    #     #     g = g.underive_all()
    #     #     print(g)
    #     #     print(g.u_size / g.l_size)
    #     #     # assert g.is_consistent
    #     #     #g.plot(with_labels=False, use_planar_drawer=False, node_size=13)
    #     #     #plt.show()
