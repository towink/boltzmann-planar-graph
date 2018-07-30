from framework.evaluation_oracle import EvaluationOracle
from framework.generic_samplers import BoltzmannSamplerBase
from framework.generic_classes import BoltzmannFrameworkError

from planar_graph_sampler.grammar.planar_graph_decomposition import planar_graph_grammar
from planar_graph_sampler.evaluations_planar_graph import all_evaluations
from planar_graph_sampler.test.dummy_grammar import dummy_sampling_grammar


def test_sampled_sizes():

    grammar = dummy_sampling_grammar()
    grammar.init()
    grammar.dummy_sampling_mode()
    #grammar.precompute_evals('G_dx_dx', 'x', 'y')

    for evaluations in all_evaluations:
        oracle = EvaluationOracle(evaluations)
        BoltzmannSamplerBase.oracle = oracle

        classes_known_dx = [
            'K',
            'K_dx',
            'K_dy',
            'J_a',

            'D',
            'D_dx',
            'S',
            'S_dx',
            'P',
            'P_dx',
            'H',
            'H_dx',

            'G_2_dx',
            'G_2_dx_dx',
            
            'G_1_dx_dx',
            'G_1',
            'G_1_dx',

            'G',
            'G_dx',
            'G_dx_dx'
        ]

        symbolic_x = [
            'x*G_1_dx(x,y)',
            'x*G_1_dx(x,y)',
            'x*G_1_dx(x,y)',
            'x*G_1_dx(x,y)',

            'x*G_1_dx(x,y)',
            'x*G_1_dx(x,y)',
            'x*G_1_dx(x,y)',
            'x*G_1_dx(x,y)',
            'x*G_1_dx(x,y)',
            'x*G_1_dx(x,y)',
            'x*G_1_dx(x,y)',
            'x*G_1_dx(x,y)',

            'x*G_1_dx(x,y)',
            'x*G_1_dx(x,y)',

            'x',
            'x',
            'x',

            'x',
            'x',
            'x'
        ]
        symbolic_y = [
            'D(x*G_1_dx(x,y),y)',
            'D(x*G_1_dx(x,y),y)',
            'D(x*G_1_dx(x,y),y)',
            'D(x*G_1_dx(x,y),y)',

            'y',
            'y',
            'y',
            'y',
            'y',
            'y',
            'y',
            'y',

            'y',
            'y',

            'y',
            'y',
            'y',

            'y',
            'y',
            'y'
        ]

        for index, label in enumerate(classes_known_dx):
            x = symbolic_x[index]
            y = symbolic_y[index]
            expected_size = oracle.get_expected_l_size(label, symbolic_x[index], symbolic_y[index])

            num_samples = 1000
            count = 0
            sizes = []
            rec_errors = 0
            while count < num_samples:
                try:
                    sizes.append(grammar.sample(label, x, y).l_size)
                    count += 1
                except RecursionError:
                    rec_errors += 1

            observed = sum(sizes) / len(sizes)

            print("class: {} \t expected: {} \t observed: {} \t rec. errors: {} \t difference: {}"
                  .format(label, expected_size, observed, rec_errors, observed/expected_size - 1))

        print()

if __name__ == "__main__":
    test_sampled_sizes()