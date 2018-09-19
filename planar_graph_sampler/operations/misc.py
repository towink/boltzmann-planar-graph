import random

import networkx as nx


def relabel_networkx(G):
    """Relabels nodes of `G` *randomly* with integers from 1 to n, the number of nodes in `G`."""
    new_labels = list(range(0, G.number_of_nodes()))
    random.shuffle(new_labels)
    relabel_dict = dict(zip(G.nodes, new_labels))
    nx.relabel_nodes(G, relabel_dict, copy=False)


def color_scale(hex_str, factor):
    """Scales a hex string by `factor`. Returns scaled hex string."""
    hex_str = hex_str.strip('#')
    if factor < 0 or len(hex_str) != 6:
        return hex_str
    r, g, b = int(hex_str[:2], 16), int(hex_str[2:4], 16), int(hex_str[4:], 16)

    def clamp(val, minimum=0, maximum=255):
        if val < minimum:
            return minimum
        if val > maximum:
            return maximum
        return int(val)

    r = clamp(r * factor)
    g = clamp(g * factor)
    b = clamp(b * factor)

    return "#%02x%02x%02x" % (r, g, b)
