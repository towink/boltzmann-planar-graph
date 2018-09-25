from timeit import default_timer as timer
from scipy.stats import chisquare

from framework.generic_samplers import BoltzmannSamplerBase, BoltzmannFrameworkError
from framework.evaluation_oracle import EvaluationOracle
from framework.generic_classes import SetClass
from planar_graph_sampler.combinatorial_classes.half_edge_graph import HalfEdgeGraph
from planar_graph_sampler.grammar.one_connected_decomposition import one_connected_graph_grammar
from planar_graph_sampler.evaluations_planar_graph import *
from planar_graph_sampler.grammar.planar_graph_decomposition import planar_graph_grammar, comps_to_nx_graph


def are_equal(g1, g2):
    """Checks if networkx graphs are equal in terms of the labels on their nodes."""
    for e in g1.edges:
        if not g2.has_edge(*e):
            return False
    for e in g2.edges:
        if not g1.has_edge(*e):
            return False
    return True


def sample_graphs_and_count(sampled_class, x, y, counting_seq, offset=1, factor=5):
    """Samples and graphs of given sizes and counts them."""
    # Make evals hard coded for now - we are going to apply this test to small sizes only anyway.
    oracle = EvaluationOracle(my_evals_10)
    BoltzmannSamplerBase.oracle = oracle
    grammar = planar_graph_grammar()
    grammar.init()
    grammar.precompute_evals(sampled_class, x, y)

    # The result will be a dictionary of graph count dicts keyed by number of nodes.
    result = {offset + i: {} for i in range(0, len(counting_seq))}

    sample_count = len(counting_seq) * [0]
    target_count = [factor * i for i in counting_seq]

    start = timer()

    while any(map(lambda pair: pair[0] < pair[1], zip(sample_count, target_count))):
        obj = grammar.sample_iterative(sampled_class).underive_all()
        if isinstance(obj, SetClass):
            obj = SetClass([he_graph.underive_all() for he_graph in obj])
        n = obj.l_size
        if offset <= n < offset + len(counting_seq):
            if isinstance(obj, SetClass):
                G = comps_to_nx_graph(obj)
            else:
                assert isinstance(obj, HalfEdgeGraph)
                G = obj.to_networkx_graph(relabel=True)
            G_found = False
            for G_key in result[n]:
                if are_equal(G, G_key):
                    result[n][G_key] += 1
                    G_found = True
            if not G_found:
                result[n][G] = 1
            sample_count[n - offset] += 1

    end = timer()
    print("time: {}".format(end - start))

    return result


def test_distribution_chi_square(significance=0.01, plot=False):
    """Tests uniformity of distribution for some small sizes using a chi-square test
    with significance level 0.01.

    If this test fails we can conclude with probability 99% that the sample does not come from a uniform distribution.
    """
    # Define a bunch of experiments.
    args_array = [

        # A066537 in the OEIS.
        # ('G', 'x', 'y', [2, 8, 64], 2),
        # ('G_dx', 'x', 'y', [2, 8, 64], 2),
        # ('G_dx_dx', 'x', 'y', [2, 8, 64], 2, 10),
        # ('G_dx_dx_dx', 'x', 'y', [8, 64], 3, 10),  # TODO FAILS!!

        # A096332 in the OEIS.
        ('G_1_dx', 'x', 'y', [4, 38], 3),
        ('G_1_dx_dx', 'x', 'y', [4, 38], 3),
        ('G_1_dx_dx_dx', 'x', 'y', [4, 38], 3),

        # A096331 in the OEIS.
        # ('G_2_dx', 'x*G_1_dx(x,y)', 'y', [10, 237], 4),
        ('G_2_dx', 'x*G_1_dx(x,y)', 'y', [10], 4, 10),
        # ('G_2_dx_dx', 'x*G_1_dx(x,y)', 'y', [10, 237], 4),
        ('G_2_dx_dx', 'x*G_1_dx(x,y)', 'y', [10], 4, 10),
        # ('G_2_dx_dx_dx', 'x*G_1_dx(x,y)', 'y', [10, 237], 4)
        ('G_2_dx_dx_dx', 'x*G_1_dx(x,y)', 'y', [10], 4, 10)
    ]

    for args in args_array:
        graph_counts = sample_graphs_and_count(*args)
        for n in graph_counts:
            print(list(graph_counts[n].values()))
            _, pvalue = chisquare(list(graph_counts[n].values()))
            try:
                assert pvalue > significance, (pvalue, *args)
            except AssertionError:
                if not plot:
                    raise
                else:
                    import matplotlib.pyplot as plt
                    import networkx as nx
                    G = min(graph_counts[n], key=graph_counts[n].get)
                    nx.draw(G, with_labels=True)
                    plt.show()


if __name__ == "__main__":
    test_distribution_chi_square(plot=True)
