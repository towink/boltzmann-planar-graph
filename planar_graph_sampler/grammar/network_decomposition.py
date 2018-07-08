from framework.generic_samplers import *
from framework.decomposition_grammar import DecompositionGrammar, AliasSampler
from framework.utils import Counter

from planar_graph_sampler.grammar.three_connected_decomposition import three_connected_graph_grammar
from planar_graph_sampler.combinatorial_classes.network import NetworkClass
from planar_graph_sampler.combinatorial_classes.halfedge import HalfEdge
from planar_graph_sampler.bijections.network_merge_in_series import NetworkMergeInSeries
from planar_graph_sampler.bijections.network_paralel_merge import NetworkMergeInParallel

counter = Counter()  # TODO does this work?


def _create_root_network_edge():
    # Create the zero-pole of the network
    root_half_edge = HalfEdge()
    root_half_edge.next = root_half_edge
    root_half_edge.prior = root_half_edge
    root_half_edge.node_nr = next(counter)

    # Creates the inf-pole
    root_half_edge_opposite = HalfEdge()
    root_half_edge_opposite.next = root_half_edge_opposite
    root_half_edge_opposite.prior = root_half_edge_opposite
    root_half_edge_opposite.node_nr = next(counter)

    # Link the poles
    root_half_edge.opposite = root_half_edge_opposite
    root_half_edge_opposite.opposite = root_half_edge

    return root_half_edge


def bij_u_atom_to_network(decomp):
    vertices_list = []
    edges_list = []
    root_half_edge = _create_root_network_edge()
    # The root edge is part from the edges list.
    edges_list.append(root_half_edge)
    result = NetworkClass(vertices_list, edges_list, root_half_edge)
    assert (decomp.get_u_size() == result.get_u_size())
    assert (decomp.get_l_size() == result.get_l_size())
    return result


def bij_s_decomp_to_network(decomp):
    # decomp has this structure: ((first_network, l_atom), second_network)
    network = decomp.first.first
    network_for_plugging = decomp.second

    # Use the serial merge bijection to merge the networks
    result = NetworkMergeInSeries().merge_networks_in_series(network, network_for_plugging)

    # Check the properties
    assert (decomp.get_u_size() == result.get_u_size())
    assert (decomp.get_l_size() == result.get_l_size())
    return result


def bij_p_decomp1_to_network(decomp):
    # decomp has the structure: (link_graph, [set of networks])
    # get the parallel composition of the networks
    result = bij_p_decomp2_to_network(decomp.second)
    # add link between the poles
    result.edges_list.append(result.root_half_edge)
    assert (decomp.get_u_size() == result.get_u_size())
    assert (decomp.get_l_size() == result.get_l_size())
    return result


def bij_p_decomp2_to_network(decomp):
    # decomp has the structure: [set of networks]
    networks = decomp

    # Merge the netwowrks in parallel
    result = networks[0]
    for i in range(1, len(networks)):
        result = NetworkMergeInParallel().merge_networks_in_parallel(result, networks[i])

    # Check the properties
    assert (decomp.get_u_size() == result.get_u_size())
    assert (decomp.get_l_size() == result.get_l_size())
    return result


def bij_g_3_arrow_to_network(decomp):
    three_connected_rooted_planar_graph = decomp.first
    # Extract the components from the three connected rooted planar graph
    vertices_list = list(three_connected_rooted_planar_graph.vertices_list)
    edges_list = list(three_connected_rooted_planar_graph.edges_list)
    root_half_edge = three_connected_rooted_planar_graph.root_half_edge
    # Create and return the network.
    return NetworkClass(vertices_list, edges_list, root_half_edge)


def network_grammar():
    """

    :return:
    """

    L = LAtomSampler
    U = UAtomSampler
    G_3_arrow = AliasSampler('G_3_arrow')
    G_3_arrow_dx = AliasSampler('G_3_arrow_dx')
    G_3_arrow_dy = AliasSampler('G_3_arrow_dy')
    Link = AliasSampler('Link')
    D = AliasSampler('D')
    D_dx = AliasSampler('D_dx')
    S = AliasSampler('S')
    S_dx = AliasSampler('S_dx')
    P = AliasSampler('P')
    P_dx = AliasSampler('P_dx')
    H = AliasSampler('H')
    H_dx = AliasSampler('H_dx')
    Bij = BijectionSampler

    grammar = DecompositionGrammar()
    grammar.add_rules(three_connected_graph_grammar().get_rules())

    grammar.add_rules({

        # networks

        'Link': Bij(U(), bij_u_atom_to_network),  # introduce this just for readability

        'D': Link + S + P + H,

        'S': Bij((Link + P + H) * L() * D, bij_s_decomp_to_network),

        'P': Bij((Link * SetSampler(1, S + H)), bij_p_decomp1_to_network)
             + Bij(SetSampler(2, S + H), bij_p_decomp2_to_network),

        'H': Bij(USubsSampler(G_3_arrow, D), bij_g_3_arrow_to_network),

        # l-derived networks

        'D_dx': S_dx + P_dx + H_dx,

        'S_dx': (P_dx + H_dx) * L() * D + (U() + P + H) * (D + L() + D_dx),  # todo bijection

        'P_dx': U() * (S_dx + H_dx) * SetSampler(0, S + H) + (S_dx + H_dx) * SetSampler(1, S + H),  # todo bijection

        'H_dx': USubsSampler(G_3_arrow_dx, D) + D_dx * USubsSampler(G_3_arrow_dy, D_dx),

    })

    return grammar


if __name__ == '__main__':
    pass
    # TODO