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

import pyboltzmann as pybo

__all__ = ['DecompositionGrammar']


def _only_if_initialized(func):
    def wrapper(self, *args, **kwargs):
        if not self._initialized:
            pybo.DecompositionGrammar._grammar_not_initialized_error()
        return func(self, *args, **kwargs)

    return wrapper


class DecompositionGrammar(object):
    """
    Represents a decomposition grammar as a collection of several rules.

    Parameters
    ----------
    rules : dict, optional (default=None)
        Initial set of rules of this grammar.
    """

    def __init__(self, rules=None):
        if rules is None:
            rules = {}
        self._initialized = False
        self._rules = rules
        self._recursive_rules = None
        self._restart_flag = False
        self._target_rule = None
        self._target_x = None
        self._target_y = None
        self._needed_oracle_queries = None

    @staticmethod
    def _grammar_not_initialized_error():
        msg = "Grammar not initialized"
        raise pybo.PyBoltzmannError(msg)

    @staticmethod
    def _missing_rule_error(alias):
        msg = "Not a rule in the grammar: {}".format(alias)
        raise pybo.PyBoltzmannError(msg)

    def init(self, target_rule, x='x', y='y'):
        """Initializes the grammar.

        A grammar can only be used for sampling after initialization.
        The initialization takes time linear in the size of the grammar.

        Parameters
        ----------
        target_rule : str
            Rule to be sampled from (sampling will also be possible for all
            rules the target depends on).
        x : str, optional (default='x')
            Symbolic x-argument.
        y : str, optional (default='y')
            Symbolic y-argument.
        """
        if target_rule not in self.rules:
            self._missing_rule_error(target_rule)
        self._target_rule, self._target_x, self._target_y = target_rule, x, y
        # Initialize the alias samplers, i.e. set their referenced grammar to
        # this grammar.
        self._init_alias_samplers()
        # Find out which rules are recursive.
        self._find_recursive_rules()
        # Automatically set target class labels of transformation samplers
        # where possible.
        self._infer_target_class_labels()
        # Get the needed oracle queries and check if they are present in the
        # oracle.
        self._initialized = True
        self._needed_oracle_queries = self._collect_oracle_queries()
        if not pybo.BoltzmannSamplerBase.oracle.contains_all(
                self._needed_oracle_queries):
                raise pybo.PyBoltzmannError(
                    "The following evals are needed: {}".format(
                        self._needed_oracle_queries
                    ))
        # Precompute the evaluations for all intermediate classes.
        self._precompute_evals()

    def _init_alias_samplers(self):
        """Sets the grammar in the alias samplers."""

        def apply_to_each(sampler):
            if isinstance(sampler, pybo.AliasSampler):
                sampler.grammar = self
                sampler._referenced_sampler = self[sampler.sampled_class]
                sampler.children = sampler._referenced_sampler,  # 1-tuple.

        for alias in self._rules:
            v = self._DFSVisitor(apply_to_each)
            self[alias].accept(v)

    def _find_recursive_rules(self):
        """Analyses the grammar to find out which rules are recursive and saves
        them.
        """
        rec_rules = []
        for alias in self.rules:
            sampler = self[alias]

            def apply_to_each(s):
                if isinstance(s, pybo.AliasSampler) \
                        and s.sampled_class == alias:
                    return alias

            v = self._DFSVisitor(apply_to_each)
            sampler.accept(v)
            if v.result:
                rec_rules += v.result
        # Duplicates might occur.
        self._recursive_rules = set(rec_rules)

    def _infer_target_class_labels(self):
        """Automatically tries to infer class labels if they are not given
        explicitly.
        """
        for alias in self.rules:
            sampler = self[alias]
            # Looks if the top sampler of a rule is a transformation sampler
            # and this case automatically sets the label of the target class.
            while isinstance(sampler, pybo.BijectionSampler):
                sampler = sampler.get_children()[0]
            if isinstance(sampler, pybo.TransformationSampler):
                sampler.sampled_class = alias

    def _collect_oracle_queries(self):
        """Returns all oracle queries that may be needed when sampling from
        the rule identified by alias.
        """
        visitor = self._CollectOracleQueriesVisitor(self._target_x,
                                                    self._target_y)
        self[self._target_rule].accept(visitor)
        return sorted(visitor.result)

    def _precompute_evals(self):
        """Precomputes all evaluations needed for sampling from the given
        class with the symbolic x and y values.
        """
        visitor = self._PrecomputeEvaluationsVisitor(self._target_x,
                                                     self._target_y)
        self[self._target_rule].accept(visitor)

    def restart_sampler(self):
        """Restarts the iterative sampler."""
        self._restart_flag = True

    def set_builder(self, rules=None, builder=pybo.DefaultBuilder()):
        """Sets a builder for a given set of rules.

        Parameters
        ----------
        rules : str or iterable, optional (default=all rules in the grammar)
            Rules for which the builder should be set.
        builder : CombinatorialClassBuilder, optional (default=DefaultBuilder)
            The builder object itself.

        Returns
        -------
        v : SetBuilderVisitor
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
        alias : str
        sampler : BoltzmannSamplerBase
        """
        self._rules[alias] = sampler

    def __setitem__(self, key, value):
        """Shorthand for add_rule."""
        self.add_rule(key, value)

    def get_rule(self, alias):
        """Returns the rule corresponding to the given alias.

        Parameters
        ----------
        alias : str

        Returns
        -------
        BoltzmannSamplerBase
        """
        return self._rules[alias]

    def __getitem__(self, item):
        """A shorthand for get_rule.

        Parameters
        ----------
        item : str

        Returns
        -------
        BoltzmannSamplerBase
        """
        return self._rules[item]

    @property
    def rules(self):
        """Gets the rules in the grammar.

        Returns
        -------
        rules : dict
        """
        return self._rules

    @rules.setter
    def rules(self, rules):
        """Adds the given set of rules.

        Parameters
        ----------
        rules : dict
            The rules to be added.
        """
        # TODO Remove, take add_rules instead.
        for alias in rules:
            self[alias] = rules[alias]

    def add_rules(self, rules):
        """Adds the given set of rules.

        Parameters
        ----------
        rules : dict
            The rules to be added.
        """
        for alias in rules:
            self[alias] = rules[alias]

    @property
    @_only_if_initialized
    def recursive_rules(self):
        """Gets the recursive rules in this grammar.

        May only be called after initialization.

        Returns
        -------
        rules : set of str
            The aliases of all recursive rules in this grammar.
        """
        return sorted(self._recursive_rules)

    @_only_if_initialized
    def is_recursive_rule(self, alias):
        """Checks if the rule corresponding to the given alias is recursive.

        Parameters
        ----------
        alias: str

        Returns
        -------
        is_recursive : bool
        """
        return alias in self._recursive_rules

    @_only_if_initialized
    def dummy_sampling_mode(self, delete_transformations=False):
        """Changes the state of the grammar to the dummy sampling mode.

        A dummy object only records its size but otherwise has no internal
        structure which is useful for testing sizes. This will overwrite
        existing builder information. After this operation, dummies can be
        sampled with sample(...).
        """
        # TODO Check if it actually works, delete otherwise.
        raise NotImplementedError
        v = self.set_builder(builder=pybo.DummyBuilder())
        if v.overwrites_builder and Settings.debug_mode:
            warnings.warn(
                "dummy_sampling_mode has overwritten existing builders")

        if delete_transformations:
            def apply_to_each(sampler):
                if isinstance(sampler, pybo.TransformationSampler) \
                        and not isinstance(sampler, pybo.RejectionSampler):
                    sampler.transformation = lambda x: x

            for alias in self.rules:
                v = self._DFSVisitor(apply_to_each)
                self[alias].accept(v)

    @_only_if_initialized
    def sample_iterative(self, alias):
        """Samples from the rule identified by `alias` in an iterative manner.

        Parameters
        ----------
        alias : str
            The rule to be sampled from

        Traverses the decomposition tree in post-order.
        The tree may be arbitrarily large and is expanded on the fly.
        """
        try:
            sampler = self[alias]
        except KeyError:
            DecompositionGrammar._missing_rule_error(alias)
        return pybo.IterativeSampler(sampler, self).sample()

    class _DFSVisitor:
        """
        Traverses the sampler hierarchy with a DFS.

        Parameters
        ----------
        f : function
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
            if isinstance(sampler, pybo.AliasSampler):
                if sampler.sampled_class in self._seen_alias_samplers:
                    # Do not recurse into the alias sampler because we have
                    # already seen it.
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
        """

        def __init__(self, builder):
            self._builder = builder
            self._overwrites_builder = False

        def visit(self, sampler):
            if isinstance(sampler, pybo.AliasSampler):
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
        Parameters.
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
            if isinstance(sampler, pybo.LSubsSampler):
                self._stack_x.append((sampler.rhs, self._x))
                self._x = sampler.rhs.oracle_query_string(self._x, self._y)
            if isinstance(sampler, pybo.USubsSampler):
                self._stack_y.append((sampler.rhs, self._y))
                self._y = sampler.rhs.oracle_query_string(self._x, self._y)
            if isinstance(sampler, pybo.AliasSampler):
                if sampler.sampled_class in self._seen_alias_samplers:
                    # Indicate that the recursion should not go further down
                    # here as we have already seen the alias.
                    return False
                else:
                    self._seen_alias_samplers.add(sampler.sampled_class)
                    # Recurse further down.
                    return True

    class _CollectOracleQueriesVisitor:
        """
        Predicts which oracle queries will be needed when sampling from the
        grammar.
        """

        def __init__(self, x, y):
            self._seen_alias_samplers = set()
            self.result = set()
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
            if isinstance(sampler, pybo.LAtomSampler) \
                    or isinstance(sampler, pybo.UAtomSampler):
                self.result.add(sampler.oracle_query_string(self._x, self._y))
            if isinstance(sampler, pybo.LSubsSampler):
                self._stack_x.append((sampler.rhs, self._x))
                self._x = sampler.rhs.oracle_query_string(self._x, self._y)
            if isinstance(sampler, pybo.USubsSampler):
                self._stack_y.append((sampler.rhs, self._y))
                self._y = sampler.rhs.oracle_query_string(self._x, self._y)
            if isinstance(sampler, pybo.TransformationSampler) \
                    and not isinstance(sampler, pybo.BijectionSampler):
                if sampler._eval_transform is None:
                    self.result.add(
                        sampler.oracle_query_string(self._x, self._y))
                # Otherwise the sampler has an eval_transform function and does
                # not directly query the oracle.
            if isinstance(sampler, pybo.AliasSampler):
                if sampler.sampled_class in self._seen_alias_samplers:
                    if sampler.is_recursive:
                        self.result.add(
                            sampler.oracle_query_string(self._x, self._y))
                    # Indicate that the recursion should not go further down
                    # here as we have already seen the alias.
                    return False
                else:
                    self._seen_alias_samplers.add(sampler.sampled_class)
                    # Recurse further down.
                    return True
