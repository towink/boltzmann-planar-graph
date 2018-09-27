import cProfile
import pstats
import random

from planar_graph_sampler.grammar.binary_tree_decomposition import binary_tree_grammar
from planar_graph_sampler.grammar.irreducible_dissection_decomposition import irreducible_dissection_grammar

# random.seed(2)

from pyboltzmann.evaluation_oracle import EvaluationOracle
from pyboltzmann.generic_samplers import BoltzmannSamplerBase
from pyboltzmann.utils import boltzmann_framework_random_gen
from planar_graph_sampler.evaluations_planar_graph import *

from planar_graph_sampler.grammar.one_connected_decomposition import one_connected_graph_grammar

import sys

from planar_graph_sampler.grammar.planar_graph_decomposition import planar_graph_grammar


def run_profiler():
    oracle = EvaluationOracle(my_evals_10000)
    BoltzmannSamplerBase.oracle = oracle
    BoltzmannSamplerBase.debug_mode = False

    grammar = planar_graph_grammar()
    grammar.init()
    symbolic_x = 'x'
    symbolic_y = 'y'
    sampled_class = 'G_dx_dx_dx'
    # symbolic_x = 'x*G_1_dx(x,y)'
    # symbolic_y = 'D(x*G_1_dx(x,y),y)'
    # sampled_class = 'K_dx_dx'
    # print(grammar.collect_oracle_queries(sampled_class, symbolic_x, symbolic_y))
    grammar._precompute_evals(sampled_class, symbolic_x, symbolic_y)

    # random.seed(0)
    # boltzmann_framework_random_gen.seed(13)

    l_sizes = []
    i = 0
    samples = 1
    while i < samples:
        obj = grammar.sample_iterative(sampled_class, symbolic_x, symbolic_y)
        l_sizes.append(obj.l_size)
        print(obj.l_size)
        i += 1
    print()
    print("avg. size: {}".format(sum(l_sizes) / len(l_sizes)))


if __name__ == "__main__":
    # random.seed(1)
    # run_profiler()
    stats = {}
    cProfile.run('run_profiler()', 'restats')
    p = pstats.Stats('restats')
    # p.strip_dirs().sort_stats('time').print_stats(50)
    # p.strip_dirs().sort_stats('cumtime').print_stats(50)
