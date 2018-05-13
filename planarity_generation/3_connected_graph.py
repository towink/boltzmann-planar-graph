#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class ThreeConnectedGraphSampler:
    """Sampler Class for 3-Connected Planar Graphs.
    Uses the Binary Tree Sampler for sampling the 3-Connected Planar Graph.
    """

    def three_connected_graph(n, epsilon=None):
        """Sample a 3-Connected Planar Graph with size n.
        If epsilon is not None the size can be between n(1-epsilon) and n(1+epsilon)
        """
        return __three_connected_graph():

    # Corresponds to 
    def __three_connected_graph()