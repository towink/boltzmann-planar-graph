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

import warnings

from framework.settings_global import Settings
from framework.class_builder import DummyBuilder
from framework.generic_samplers import *
from framework.iterative_sampler import _IterativeSampler


class DecompositionGrammar(object):
    """
    Represents a decomposition grammar as a collection of several rules.

    Parameters
    ----------
    rules: dict, optional (default=None)
        Initial set of rules of this grammar.
    """

    def __init__(self, rules=None):
        if rules is None:
            rules = {}
        self._rules = rules
        self._recursive_rules = None
        self._restart_flag = False

    @staticmethod
    def _grammar_not_initialized_error():
        msg = "Grammar not initialized"
        raise BoltzmannFrameworkError(msg)

    @staticmethod
    def _missing_rule_error(alias):
        msg = "Not a rule in the grammar: {}".format(alias)
        raise BoltzmannFrameworkError(msg)

    def init(self):
        """Initializes the grammar.

        A grammar can only be used for sampling after initialization.
        """
        # Initialize the alias samplers (set their referenced grammar to this grammar).
        self._init_alias_samplers()
        # Find out which rules are recursive.
        self._find_recursive_rules()
        # Automatically set target class labels of transformation samplers where possible.
        self._infer_target_class_labels()

    def _init_alias_samplers(self):
        """Sets the grammar in the alias samplers."""

        def apply_to_each(sampler):
            if isinstance(sampler, AliasSampler):
                sampler.grammar = self
                sampler._referenced_sampler = self[sampler.sampled_class]
                sampler.children = sampler._referenced_sampler,  # 1-tuple.

        for alias in self._rules:
            v = self._DFSVisitor(apply_to_each)
            self[alias].accept(v)

    def _find_recursive_rules(self):
        """Analyses the grammar to find out which rules are recursive and saves them."""
        rec_rules = []
        for alias in self.rules:
            sampler = self[alias]

            def apply_to_each(sampler):
                if isinstance(sampler, AliasSampler) and sampler.sampled_class == alias:
                    return alias

            v = self._DFSVisitor(apply_to_each)
            sampler.accept(v)
            if v.result:
                rec_rules += v.result
        # Duplicates might occur.
        self._recursive_rules = set(rec_rules)

    def _infer_target_class_labels(self):
        """Automatically tries to infer class labels if they are not given explicitly."""
        for alias in self.rules:
            sampler = self[alias]
            # Looks if the top sampler of a rule is a transformation sampler and this case automatically sets the label
            # of the target class.
            while isinstance(sampler, BijectionSampler):
                sampler = sampler.get_children()[0]
            if isinstance(sampler, TransformationSampler):
                sampler.sampled_class = alias

    def restart_sampler(self):
        """Restarts the iterative sampler."""
        self._restart_flag = True

    def set_builder(self, rules=None, builder=DefaultBuilder()):
        """Sets a builder for a given set of rules.

        Parameters
        ----------
        rules: str or iterable, optional (default=all rules in the grammar)
        builder: CombinatorialClassBuilder, optional (default=DefaultBuilder)

        Returns
        -------
        v: SetBuilderVisitor
        """
        if rules is None:
            rules = self._rules.keys()
        if type(rules) is str:
            rules = [rules]
        v = self._SetBuilderVisitor(builder)
        for alias in rules:
            self[alias].accept(v)
        return v

    def add_rule(self, alias, sampler):
        """Adds a decomposition rule to this grammar.

        Parameters
        ----------
        alias: str
        sampler: BoltzmannSamplerBase
        """
        self._rules[alias] = sampler

    def __setitem__(self, key, value):
        """Shorthand for add_rule."""
        self.add_rule(key, value)

    def get_rule(self, alias):
        """Returns the rule corresponding to the given alias.

        Parameters
        ----------
        alias: str

        Returns
        -------
        BoltzmannSamplerBase
        """
        return self._rules[alias]

    def __getitem__(self, item):
        """A shorthand for get_rule.

        Parameters
        ----------
        item: str

        Returns
        -------
        BoltzmannSamplerBase
        """
        return self._rules[item]

    @property
    def rules(self):
        return self._rules

    @rules.setter
    def rules(self, rules):
        """Adds the given set of rules.

        Parameters
        ----------
        rules: dict
            The rules to be added.
        """
        for alias in rules:
            self[alias] = rules[alias]

    @property
    def recursive_rules(self):
        """Gets the recursive rules in this grammar.

        May only be called after initialization.

        Returns
        -------
        rules: set of str
            The aliases of all recursive rules in this gramamr.
        """
        if self._recursive_rules is None:
            DecompositionGrammar._grammar_not_initialized_error()
        return sorted(self._recursive_rules)

    def is_recursive_rule(self, alias):
        """Checks if the rule corresponding to the given alias is recursive.

        Parameters
        ----------
        alias: str

        Returns
        -------
        is_recursive: bool
        """
        if self._recursive_rules is None:
            DecompositionGrammar._grammar_not_initialized_error()
        return alias in self.recursive_rules

    def collect_oracle_queries(self, alias, x, y):
        """Returns all oracle queries that may be needed when sampling from the rule identified by alias.

        Parameters
        ----------
        alias: str
        x: str
            Symbolic x value.
        y: str
            Symbolic y value.

        Returns
        -------
        oracle_queries: set of str
        """
        if alias not in self.rules:
            DecompositionGrammar._missing_rule_error(alias)
        visitor = self._CollectOracleQueriesVisitor(x, y)
        self[alias].accept(visitor)
        return sorted(visitor.result)

    def precompute_evals(self, alias, x, y):
        """Precomputes all evaluations needed for sampling from the given class with the symbolic x and y values.

        Parameters
        ----------
        alias: str
        x: str
            Symbolic x value.
        y: str
            Symbolic y value.
        """
        if alias not in self.rules:
            DecompositionGrammar._missing_rule_error(alias)
        visitor = self._PrecomputeEvaluationsVisitor(x, y)
        self[alias].accept(visitor)

    def sample(self, alias, x, y):
        """Samples from the rule identified by the given alias.

        Parameters
        ----------
        alias: str
        x: str
            Symbolic x value.
        y: str
            Symbolic y value.

        Returns
        -------
        CombinatorialClass
        """
        sampler = None
        try:
            sampler = self[alias]
        except KeyError:
            DecompositionGrammar._missing_rule_error(alias)
        return sampler.sample(x, y)

    def dummy_sampling_mode(self, delete_transformations=False):
        """Changes the state of the grammar to the dummy sampling mode.

        A dummy object only records its size but otherwise has no internal structure which is useful for testing sizes.
        This will overwrite existing builder information. After this operation, dummies can be sampled with sample(...).
        """
        v = self.set_builder(builder=DummyBuilder())
        if v.overwrites_builder and Settings.debug_mode:
            warnings.warn("dummy_sampling_mode has overwritten existing builders")

        if delete_transformations:
            def apply_to_each(sampler):
                if isinstance(sampler, TransformationSampler) and not isinstance(sampler, RejectionSampler):
                    sampler.transformation = lambda x: x

            for alias in self.rules:
                v = self._DFSVisitor(apply_to_each)
                self[alias].accept(v)

    def sample_iterative(self, alias):
        """Samples from the rule identified by `alias` in an iterative manner.

        Traverses the decomposition tree in post-order.
        The tree may be arbitrarily large and is expanded on the fly.
        """

        return _IterativeSampler(self[alias], self).sample()

    class _DFSVisitor:
        """
        Traverses the sampler hierarchy with a DFS.

        Parameters
        ----------
        f: function
            Function to be applied to all nodes (=samplers).
        """

        def __init__(self, f):
            self._f = f
            self._result = []
            self._seen_alias_samplers = set()
            self._grammar = None

        def visit(self, sampler):
            # Apply the function to the current sampler.
            r = self._f(sampler)
            # Append the return value to the result list if any.
            if r is not None:
                self.result.append(r)
            if isinstance(sampler, AliasSampler):
                if sampler.sampled_class in self._seen_alias_samplers:
                    # Do not recurse into the alias sampler because we have already seen it.
                    return False
                else:
                    self._seen_alias_samplers.add(sampler.sampled_class)
                    # Recurse further down.
                    return True

        @property
        def result(self):
            return self._result

    class _SetBuilderVisitor:
        """
        Sets builders until hitting an AliasSampler.

        Parameters
        ----------
        builder: CombinatorialClassBuilder
            The builder to be set to all visited samplers.
        """

        def __init__(self, builder):
            self._builder = builder
            self._overwrites_builder = False

        def visit(self, sampler):
            if isinstance(sampler, AliasSampler):
                # Don't recurse into the alias samplers.
                return False
            else:
                # Otherwise set the given builder.
                if sampler.builder is not None:
                    self._overwrites_builder = True
                sampler.builder = self._builder

        @property
        def overwrites_builder(self):
            return self._overwrites_builder

    class _PrecomputeEvaluationsVisitor:
        """
        Precomputes evaluations for the sampler in the given hierarchy.
        Parameters
        ----------
        x: str
            Symbolic x value.
        y: str
            Symbolic y value.
        """

        def __init__(self, x, y):
            self._seen_alias_samplers = set()
            self._x = x
            self._y = y
            self._stack_x = []
            self._stack_y = []

        def visit(self, sampler):
            if self._stack_x and self._stack_x[-1][0] == sampler:
                _, x = self._stack_x.pop()
                self._x = x
            if self._stack_y and self._stack_y[-1][0] == sampler:
                _, y = self._stack_y.pop()
                self._y = y
            sampler.precompute_eval(self._x, self._y)
            if isinstance(sampler, LSubsSampler):
                self._stack_x.append((sampler.rhs, self._x))
                self._x = sampler.rhs.oracle_query_string(self._x, self._y)
            if isinstance(sampler, USubsSampler):
                self._stack_y.append((sampler.rhs, self._y))
                self._y = sampler.rhs.oracle_query_string(self._x, self._y)
            if isinstance(sampler, AliasSampler):
                if sampler.sampled_class in self._seen_alias_samplers:
                    # Indicate that the recursion should not go further down here as we have already seen the alias.
                    return False
                else:
                    self._seen_alias_samplers.add(sampler.sampled_class)
                    # Recurse further down.
                    return True

    class _CollectOracleQueriesVisitor:
        """
        Predicts which oracle queries will be needed when sampling from the grammar.
        This might still be a bit buggy but its not crucial for the correctness of the samplers.

        Parameters
        ----------
        x: str
            Symbolic x value.
        y: str
            Symbolic y value.
        """

        def __init__(self, x, y):
            self._seen_alias_samplers = set()
            self._result = set()
            self._x = x
            self._y = y
            self._stack_x = []
            self._stack_y = []

        @property
        def result(self):
            return self._result

        def visit(self, sampler):
            if self._stack_x and self._stack_x[-1][0] == sampler:
                _, x = self._stack_x.pop()
                self._x = x
            if self._stack_y and self._stack_y[-1][0] == sampler:
                _, y = self._stack_y.pop()
                self._y = y
            if isinstance(sampler, LAtomSampler) or isinstance(sampler, UAtomSampler):
                self._result.add(sampler.oracle_query_string(self._x, self._y))
            if isinstance(sampler, LSubsSampler):
                self._stack_x.append((sampler.rhs, self._x))
                self._x = sampler.rhs.oracle_query_string(self._x, self._y)
            if isinstance(sampler, USubsSampler):
                self._stack_y.append((sampler.rhs, self._y))
                self._y = sampler.rhs.oracle_query_string(self._x, self._y)
            if isinstance(sampler, TransformationSampler) and not isinstance(sampler, BijectionSampler):
                if sampler._eval_transform is None:
                    self._result.add(sampler.oracle_query_string(self._x, self._y))
                # Otherwise the sampler has an eval_transform function and does not directly query the oracle.
            if isinstance(sampler, AliasSampler):
                if sampler.sampled_class in self._seen_alias_samplers:
                    if sampler.is_recursive:
                        self._result.add(sampler.oracle_query_string(self._x, self._y))
                    # Indicate that the recursion should not go further down here as we have already seen the alias.
                    return False
                else:
                    self._seen_alias_samplers.add(sampler.sampled_class)
                    # Recurse further down.
                    return True


