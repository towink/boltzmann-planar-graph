# -*- coding: utf-8 -*-
#    Copyright (C) 2018 by
#    Tobias Winkler <tobias.winkler1@rwth-aachen.de>
#    All rights reserved.
#    BSD license.
#
# Authors:  Tobias Winkler <tobias.winkler1@rwth-aachen.de>
#

import pyboltzmann as pybo

__all__ = ['IterativeSampler']


class IterativeSampler(object):
    """
    Implements the iterative sampling mechanism.

    Parameters
    ----------
    sampler : BoltzmannSamplerBase
    grammar : DecompositionGrammar
    is_restartable : bool
    """

    def __init__(self, sampler, grammar, is_restartable=False):
        self.sampler = sampler
        self.grammar = grammar

        # self.is_restartable = True
        # self.sample = self.sample_with_restart_check
        self.is_restartable = is_restartable
        if is_restartable:
            self.sample = self.sample_with_restart_check

    class _ResultStack(list):
        """Modified stack that keeps track of the total l-size it contains."""

        def __init__(self, grammar):
            self.l_size = 0
            self.grammar = grammar
            # print()

        def append(self, obj):
            list.append(self, obj)
            self.l_size += obj.l_size
            # print(self.l_size)
            if self.l_size > 1000:
                # self.grammar.restart_sampler()
                pass

        def pop(self, **kwargs):
            obj = list.pop(self)
            self.l_size -= obj.l_size
            return obj

    def sample_with_restart_check(self):
        """Invokes the iterative sampler for the given symbolic parameters.

        In each iteration, check if the sampler should be restarted.
        This can notable decrease performance.
        """

        # Main stack.
        stack = [self.sampler]
        # Stack that holds the intermediate sampling results.
        # result_stack = self._ResultStack(self.grammar)
        result_stack = []
        # The previously visited node in the decomposition tree.
        prev = None

        while stack:

            # Check if the sampler should be restarted.
            if self.grammar._restart_flag:
                if not self.is_restartable:
                    raise PyBoltzmannError(
                        "Trying to restart a non-restartable sampler.")
                # print("Restarting ...")
                self.grammar._restart_flag = False
                stack = [self.sampler]
                # result_stack = self._ResultStack(self.grammar)
                result_stack = []
                prev = None
                continue

            # Get top of stack.
            curr = stack[-1]

            curr.sample_iterative(stack, result_stack, prev, self.grammar)

            prev = curr

        assert len(result_stack) == 1
        assert result_stack[0] is not None
        return result_stack[0]

    def sample(self):
        """Invokes the iterative sampler for the given symbolic parameters."""

        # Main stack.
        stack = [self.sampler]
        # Stack that holds the intermediate sampling results.
        # result_stack = self._ResultStack()
        result_stack = []
        # The previously visited node in the decomposition tree.
        prev = None

        while stack:

            # Get top of stack.
            curr = stack[-1]

            curr.sample_iterative(stack, result_stack, prev, self.grammar)

            prev = curr

        assert len(result_stack) == 1
        assert result_stack[0] is not None
        return result_stack[0]
