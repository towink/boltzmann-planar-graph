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

from __future__ import division

from framework.class_builder import DefaultBuilder
from framework.generic_classes import *
from framework.iterative_sampler import _IterativeSampler
from framework.utils import *


def return_precomp(func):
    def wrapper(self, *args, **kwargs):
        if self._precomputed_eval is not None:
            return self._precomputed_eval
        return func(self, *args, **kwargs)

    return wrapper


class BoltzmannSamplerBase(object):
    """
    Abstract base class for Boltzmann samplers.

    Attributes
    ----------
    oracle : EvaluationOracle
        The oracle to be used by all instantiations of this class and its
        subclasses.
    debug_mode : bool
        Will perform additional (possibly time consuming) checks when set to
        True.
    """

    oracle = None
    debug_mode = False

    __slots__ = '_builder', '_precomputed_eval', 'children'

    def __init__(self):
        # Samplers are always initialized with the default builder.
        self._builder = DefaultBuilder()
        self._precomputed_eval = None
        self.children = None

    @property
    def sampled_class(self):
        """Returns a string representation of the sampled class.

        Returns
        -------
        str
            A string representation of the sampled class.
        """
        raise NotImplementedError

    def sample_iterative(self, stack, result_stack, prev, grammar):
        """Called by the iterative sampling process.

        Do not call directly.

        Parameters
        ----------
        stack : Stack
        result_stack: Stack
        prev : BoltzmannSamplerBase
        grammar : DecompositionGrammar
        """
        raise NotImplementedError

    def eval(self, x, y):
        """Gets the evaluation of the generating function of the class being
        sampled.

        Possibly queries the oracle directly or indirectly.

        Parameters
        ----------
        x : str
            Symbolic x argument.
        y : str
            Symbolic y argument.

        Returns
        -------
        evaluation : float
            The value of the generating function of the sampled class at the
            given point.
        """
        raise NotImplementedError

    def precompute_eval(self, x, y):
        """Precomputes the evaluation of the generating function and stores it.

        Parameters
        ----------
        x : str
            Symbolic x argument.
        y : str
            Symbolic y argument.
        """
        self._precomputed_eval = self.eval(x, y)

    def oracle_query_string(self, x, y):
        """String used as key in oracle.

        This is like a 'symbolic evaluation' of the generating function of the
        class being sampled from this sampler.

        Parameters
        ----------
        x : str
            Symbolic x argument.
        y : str
            Symbolic y argument.

        Returns
        -------
        query_string : str
            String that may be used to query the oracle for the generating
            function of the sampled class.
        """
        raise NotImplementedError

    @property
    def builder(self):
        """Returns the builder registered with this sampler or `None`.

        Returns
        -------
        builder : CombinatorialClassBuilder
            Builder registered with this sampler or `None`.
        """
        return self._builder

    @builder.setter
    def builder(self, builder):
        """Registers a builder for the output class.

        Output will be in generic format if not set.

        Parameters
        ----------
        builder : CombinatorialClassBuilder
        """
        self._builder = builder

    def get_children(self):
        """Gets the samplers this sampler depends on (if applicable).

        Applies to all samplers except atom samplers.

        Returns
        -------
        iterable
            An iterable of all samplers this sampler depends on.
        """
        raise NotImplementedError

    def accept(self, visitor):
        """Accepts a visitor.

        Parameters
        ----------
        visitor : object
            The visitor to be applied.
        """
        visitor.visit(self)
        for child in self.get_children():
            child.accept(visitor)

    def __add__(self, other):
        """Sum construction of samplers.

        Parameters
        ----------
        other : BoltzmannSamplerBase
            The right hand side argument.

        Returns
        -------
        BoltzmannSamplerBase
            The sum-sampler resulting from this sampler and the given sampler.
        """
        return SumSampler(self, other)

    def __mul__(self, other):
        """Product construction of samplers.

        Parameters
        ----------
        other : BoltzmannSamplerBase
            The right hand side argument.

        Returns
        -------
        BoltzmannSamplerBase
            The product-sampler resulting from this sampler and the given
            sampler.
        """
        return ProdSampler(self, other)

    def __rmul__(self, other):
        """Adds (sum construction) a sampler to itself several times.

        Parameters
        ----------
        other : int
            The left hand side argument.

        Returns
        -------
        BoltzmannSamplerBase
            The sampler resulting from this operation.

        Notes
        -----
        This only affects the generating function but not the outcome.
        """
        return TransformationSampler(
            self,
            eval_transform=lambda evl, x, y: other * evl,
        )

    def __pow__(self, power, modulo=None):
        """Multiplies a sampler with itself.

        Parameters
        ----------
        power : int
            The right hand side argument, must be either 2 or 3.

        Returns
        -------
        BoltzmannSamplerBase
            The sampler resulting from this operation.

        """
        # TODO Support all integers.
        if power == 2:
            return self * self
        if power == 3:
            return self * self * self
        else:
            raise ValueError("Power only implemented for integers 2 and 3")


class AtomSampler(BoltzmannSamplerBase):
    """Abstract base class for atom samplers."""

    def __init__(self):
        super(AtomSampler, self).__init__()
        self.children = ()  # Empty tuple.

    @property
    def sampled_class(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def sample_iterative(self, stack, result_stack, prev, grammar):
        # Atom samplers are the leaves in the decomposition tree.
        stack.pop()
        result_stack.append(self.sample())

    @return_precomp
    def eval(self, x, y):
        return self.oracle[self.oracle_query_string(x, y)]

    def oracle_query_string(self, x, y):
        raise NotImplementedError

    def get_children(self):
        return self.children


class ZeroAtomSampler(AtomSampler):
    """Sampler for the zero atom."""

    def __init__(self):
        super(ZeroAtomSampler, self).__init__()

    @property
    def sampled_class(self):
        return '1'

    def sample(self):
        return self._builder.zero_atom()

    def eval(self, x, y):
        # See 3.2, instead of querying '1' to the oracle we just return it.
        return 1

    def oracle_query_string(self, x, y):
        return '1'


class LAtomSampler(AtomSampler):
    """Sampler for the l-atom (labeled atom)."""

    def __init__(self):
        super(LAtomSampler, self).__init__()

    @property
    def sampled_class(self):
        return 'L'

    def sample(self):
        return self._builder.l_atom()

    def oracle_query_string(self, x, y):
        # Generating function of the l-atom is just x.
        return x


class UAtomSampler(AtomSampler):
    """Sampler for the u-atom (unlabeled atom)."""

    def __init__(self):
        super(UAtomSampler, self).__init__()

    @property
    def sampled_class(self):
        return 'U'

    def sample(self):
        return self._builder.u_atom()

    def oracle_query_string(self, x, y):
        # Generating function of the u-atom is just y.
        return y


class BinarySampler(BoltzmannSamplerBase):
    """
    Abstract base class for a sampler that depends on exactly 2 other samplers.

    Parameters
    ----------
    lhs: BoltzmannSamplerBase
        The left hand side argument.
    rhs: BoltzmannSamplerBase
        The right hand side argument.
    op_symbol: str
        The string representation of the implemented operator.
    """
    __slots__ = 'lhs', 'rhs', '_op_symbol'

    def __init__(self, lhs, rhs, op_symbol):
        super(BinarySampler, self).__init__()
        self.lhs = lhs
        # If the arguments are the same object then we make a copy to be able
        # distinguish them in the iterative sampler.
        if lhs is rhs:
            from copy import copy
            rhs = copy(lhs)
        self.rhs = rhs
        self._op_symbol = op_symbol
        self.children = self.lhs, self.rhs

    @property
    def sampled_class(self):
        return "({}{}{})".format(
            self.lhs.sampled_class, self._op_symbol, self.rhs.sampled_class)

    def sample_iterative(self, stack, result_stack, prev, grammar):
        raise NotImplementedError

    def eval(self, x, y):
        raise NotImplementedError

    def oracle_query_string(self, x, y):
        return "({}{}{})".format(
            self.lhs.oracle_query_string(x, y),
            self._op_symbol,
            self.rhs.oracle_query_string(x, y)
        )

    def get_children(self):
        return self.lhs, self.rhs


class SumSampler(BinarySampler):
    """Samples from the disjoint union of the two underlying classes."""

    __slots__ = 'prob_pick_lhs'

    def __init__(self, lhs, rhs):
        super(SumSampler, self).__init__(lhs, rhs, '+')

    @return_precomp
    def eval(self, x, y):
        # For the sum class (disjoint union) the generating function is the sum
        # of the two generating functions.
        return self.lhs.eval(x, y) + self.rhs.eval(x, y)

    def sample_iterative(self, stack, result_stack, prev, grammar):
        if prev is None or self in prev.children:
            if bern(self.lhs._precomputed_eval / self._precomputed_eval):
                stack.append(self.lhs)
            else:
                stack.append(self.rhs)
        else:
            stack.pop()


class ProdSampler(BinarySampler):
    """Samples from the cartesian product of the two underlying classes."""

    def __init__(self, lhs, rhs):
        super(ProdSampler, self).__init__(lhs, rhs, '*')

    @return_precomp
    def eval(self, x, y):
        # For the product class (cartesian prod.) the generating function is
        # the product of the two generating functions.
        return self.lhs.eval(x, y) * self.rhs.eval(x, y)

    def oracle_query_string(self, x, y):
        return "{}{}{}".format(
            self.lhs.oracle_query_string(x, y),
            self._op_symbol,
            self.rhs.oracle_query_string(x, y)
        )

    def sample_iterative(self, stack, result_stack, prev, grammar):
        if prev is None or self in prev.children:
            stack.append(self.lhs)
        elif self.lhs is prev:
            # Left child has already been visited - visit right child.
            stack.append(self.rhs)
        else:
            # Both children have been processed.
            stack.pop()
            # A bit faster than popping twice.
            # arg_lhs, arg_rhs = result_stack[-2:]
            # del result_stack[-2:]
            arg_rhs = result_stack.pop()
            arg_lhs = result_stack.pop()
            result_stack.append(self.builder.product(arg_lhs, arg_rhs))


class LSubsSampler(BinarySampler):
    """Samples from the class resulting from substituting the l-atoms of lhs
    with objects of rhs."""

    def __init__(self, lhs, rhs):
        super(LSubsSampler, self).__init__(lhs, rhs, ' lsubs ')

    @return_precomp
    def eval(self, x, y):
        return self.lhs.eval(self.rhs.oracle_query_string(x, y), y)

    def oracle_query_string(self, x, y):
        #  A(B(x,y),y) where A = lhs and B = rhs (see 3.2).
        return self.lhs.oracle_query_string(
            self.rhs.oracle_query_string(x, y), y)

    def sample_iterative(self, stack, result_stack, prev, grammar):
        if prev is None or self in prev.get_children():
            # Sample from lhs with substituted x.
            stack.append(self.lhs)
        else:
            stack.pop()
            # Get the object in which the l-atoms have to be replaced from the
            # result stack.
            core_object = result_stack.pop()
            # Replace the atoms and push result.
            sampler = _IterativeSampler(self.rhs, grammar)
            res = core_object.replace_l_atoms(sampler)  # Recursion for now.
            result_stack.append(res)


class USubsSampler(BinarySampler):
    """Samples from the class resulting from substituting the u-atoms of lhs
    with objects of rhs."""

    def __init__(self, lhs, rhs):
        super(USubsSampler, self).__init__(lhs, rhs, ' usubs ')

    @return_precomp
    def eval(self, x, y):
        return self.lhs.eval(x, self.rhs.oracle_query_string(x, y))

    def oracle_query_string(self, x, y):
        # A(x,B(x,y)) where A = lhs and B = rhs (see 3.2).
        return self.lhs.oracle_query_string(
            x, self.rhs.oracle_query_string(x, y))

    def sample_iterative(self, stack, result_stack, prev, grammar):
        if prev is None or self in prev.children:
            stack.append(self.lhs)
        else:
            stack.pop()
            # Get the object in which the u-atoms have to be replaced from the
            # result stack.
            core_object = result_stack.pop()
            # Recover the old y from the stack.
            # y = result_stack.pop()
            # Replace the atoms and push result.
            sampler = _IterativeSampler(self.rhs, grammar)
            res = core_object.replace_u_atoms(sampler)  # Recursion for now.
            result_stack.append(res)


class UnarySampler(BoltzmannSamplerBase):
    """
    Abstract base class for samplers that depend exactly on one other sampler.

    Parameters
    ----------
    sampler : BoltzmannSamplerBase
        The sampler this sampler depends on.
    """

    __slots__ = '_sampler'

    def __init__(self, sampler):
        super(UnarySampler, self).__init__()
        self._sampler = sampler
        self.children = sampler,  # 1-tuple.

    @property
    def sampled_class(self):
        raise NotImplementedError

    def sample_iterative(self, stack, result_stack, prev, grammar):
        raise NotImplementedError

    def eval(self, x, y):
        raise NotImplementedError

    def oracle_query_string(self, x, y):
        raise NotImplementedError

    def get_children(self):
        return self.children


class SetSampler(UnarySampler):
    """
    Samples a set of elements from the underlying class.

    A set has no order and no duplicate elements. The underlying class may not
    contain objects with l-size 0. Otherwise this sampler will not output
    correct results!

    Parameters
    ----------
    d : int
        The minimum size of the set.
    sampler : BoltzmannSamplerBase
        The sampler that produces the elements in the set.

    Notes
    -----
    It is not checked/ensured automatically that objects from the underlying
    class do not have l-size 0.
    """

    __slots__ = '_d'

    def __init__(self, d, sampler):
        super(SetSampler, self).__init__(sampler)
        self._d = d

    @property
    def sampled_class(self):
        return "Set_{}({})".format(self._d, self._sampler.sampled_class)

    @return_precomp
    def eval(self, x, y):
        # The generating function of a set class is a tail of the exponential
        # row evaluated at the generating function of the underlying class
        # (see 3.2).
        return exp_tail(self._d, self._sampler.eval(x, y))

    def oracle_query_string(self, x, y):
        return "exp_{}({})".format(
            self._d, self._sampler.oracle_query_string(x, y))

    def _draw_k(self, x, y):
        return pois(self._d, self._sampler.eval(x, y))

    def sample_iterative(self, stack, result_stack, prev, grammar):
        # We use recursion here for now.
        stack.pop()
        set_elems_sampler = self._sampler
        k = pois(self._d, self._sampler._precomputed_eval)
        sampler = _IterativeSampler(set_elems_sampler, grammar)
        set_elems = []
        for _ in range(k):
            obj = sampler.sample()
            set_elems.append(obj)
        result_stack.append(self.builder.set(set_elems))


class TransformationSampler(UnarySampler):
    """
    Generic sampler that transforms the the objects sampled from the base class
    to a new class.

    For bijections use the base class BijectionSampler.

    Parameters
    ----------
    sampler : BoltzmannSamplerBase
    f : transformation function, optional (default=id)
    eval_transform : generating function transformation, optional (default=id)
    target_class_label : str, optional (default=None)
    """

    def __init__(self, sampler, f=None, eval_transform=None,
                 target_class_label=None):
        super(TransformationSampler, self).__init__(sampler)
        self.f = f
        self._eval_transform = eval_transform
        if target_class_label is None:
            # Set a default label for the target class based on the name of the
            # transformation function.
            if f is not None:
                self._target_class_label = "{}({})".format(
                    f.__name__, sampler.sampled_class)
            else:
                self._target_class_label = sampler.sampled_class
        else:
            self._target_class_label = target_class_label

    @property
    def sampled_class(self):
        return self._target_class_label

    @sampled_class.setter
    def sampled_class(self, label):
        self._target_class_label = label

    @return_precomp
    def eval(self, x, y):
        # If a transformation of the generating function is given, apply it.
        if self._eval_transform is not None:
            return self._eval_transform(self._sampler.eval(x, y), x, y)
        # Otherwise query the oracle because we cannot infer the evaluation in
        # the case of a general transformation.
        return self.oracle.get(self.oracle_query_string(x, y))

    def oracle_query_string(self, x, y):
        return "{}({},{})".format(self.sampled_class, x, y)

    def sample_iterative(self, stack, result_stack, prev, grammar):
        if prev is None or self in prev.children:
            stack.append(self._sampler)
        else:
            stack.pop()
            to_transform = result_stack.pop()
            # A transformation may not be given.
            if self.f is not None:
                result_stack.append(self.f(to_transform))
            else:
                result_stack.append(to_transform)


class BijectionSampler(TransformationSampler):
    """
    Samples a class that is isomorphic to the underlying class.

    Parameters
    ----------
    sampler : BoltzmannSamplerBase
    f : transformation function
    target_class_label : str, optional (default=None)
    """

    def __init__(self, sampler, f, target_class_label=None):
        super(BijectionSampler, self).__init__(sampler, f, None,
                                               target_class_label)

    @return_precomp
    def eval(self, x, y):
        # Since the target class is isomorphic to the underlying class, the
        # evaluation is also the same.
        return self._sampler.eval(x, y)

    def oracle_query_string(self, x, y):
        return self._sampler.oracle_query_string(x, y)


class HookSampler(BijectionSampler):
    """
    Special bijection used to inject arbitrary code into the sampling process.

    Code could for example modify global settings.

    Parameters
    ----------
    before : function
        Code executed before the underlying sampler is called.
    after : function, optional (default=None)
        Code executed after the underlying sampler has terminated.
    """

    def __init__(self, sampler, before, after=None):
        super(HookSampler, self).__init__(sampler, None, None)
        self.before = before
        self.after = after

    def sample_iterative(self, stack, result_stack, prev, grammar):
        if prev is None or self in prev.children:
            stack.append(self._sampler)
            # Execute the before-hook.
            self.before()
        else:
            stack.pop()
            # Execute the after-hook.
            if self.after is not None:
                self.after()


class RestartableSampler(BijectionSampler):
    """Wrapper to indicate that a sampler can be restarted."""

    def __init__(self, sampler):
        super(RestartableSampler, self).__init__(sampler, None, None)

    def sample_iterative(self, stack, result_stack, prev, grammar):
        stack.pop()
        wrapped_sampler = self._sampler
        restartable_sampler = _IterativeSampler(wrapped_sampler, grammar,
                                                is_restartable=True)
        obj = restartable_sampler.sample()
        result_stack.append(obj)


class RejectionSampler(TransformationSampler):
    """
    Generic rejection sampler, special case of transformation.

    Parameters
    ----------
    sampler : BoltzmannSamplerBase
        Sampler of the underlying class.
    is_acceptable : function
        Criterion for accepting an object from the underlying class.
    eval_transform : function, optional (default=None)
        Optional transformation of the evaluation function.
    target_class_label : str, optional (default=None)
        Optional label of the sampled class.
    """

    def __init__(self, sampler, is_acceptable, eval_transform=None,
                 target_class_label=None):
        super(RejectionSampler, self).__init__(
            sampler, is_acceptable, eval_transform, target_class_label)
        self._rejections_count = 0

    @property
    def rejections_count(self):
        """Counts the number of unsuccessful sampling operations.

        Returns
        -------
        rejections_count : int
            Number of rejections in the last sampling operation.
        """
        raise NotImplementedError

    def sample_iterative(self, stack, result_stack, prev, grammar):
        if prev is None or self in prev.children:
            stack.append(self._sampler)
        else:
            obj_to_check = result_stack.pop()
            is_acceptable = self.f
            if is_acceptable(obj_to_check):
                stack.pop()
                result_stack.append(obj_to_check)
            else:
                stack.append(self._sampler)


class UDerFromLDerSampler(TransformationSampler):
    """
    Samples the u-derived (dy) class of the given l-derived (dx) class sampler.

    Parameters
    ----------
    sampler : BoltzmannSamplerBase
        A sampler of the l-derived class. Must sample an LDerivedClass.
    alpha_u_l : float
        Limit value of u-size/l-size of the underlying class.
    """

    def __init__(self, sampler, alpha_u_l):
        # Try to infer a label that makes sense.
        if sampler.sampled_class[-2:] is 'dx':
            label = "{}dy".format(sampler.sampled_class[:-2])
        else:
            label = "{}_dy_from_dx".format(sampler.sampled_class)
        super(UDerFromLDerSampler, self).__init__(sampler, None, None, label)
        self._alpha_u_l = alpha_u_l

    def sample_iterative(self, stack, result_stack, prev, grammar):
        if prev is None or self in prev.children:
            stack.append(self._sampler)
        else:
            obj_to_check = result_stack.pop()

            def is_acceptable(gamma):
                # See Lemma 6.
                return bern(
                    (1 / self._alpha_u_l) * (gamma.u_size / (gamma.l_size + 1))
                )

            if is_acceptable(obj_to_check):
                stack.pop()
                result_stack.append(
                    UDerivedClass(obj_to_check.base_class_object))
            else:
                stack.append(self._sampler)


class LDerFromUDerSampler(TransformationSampler):
    """
    Samples the l-derived (dx) class of the given u-derived (dy) class.

    Parameters
    ----------
    sampler : BoltzmannSamplerBase
        A sampler of the u-derived class. Must sample a UDerivedClass.
    alpha_l_u : float
        Limit value of l-size/u-size of the underlying class.
    """

    def __init__(self, sampler, alpha_l_u):
        # Try to infer a label that makes sense.
        if sampler.sampled_class[-2:] is 'dy':
            label = "{}dx".format(sampler.sampled_class[:-2])
        else:
            label = "{}_dx_from_dy".format(sampler.sampled_class)
        super(LDerFromUDerSampler, self).__init__(sampler, None, None, label)
        self._alpha_l_u = alpha_l_u

    def sample_iterative(self, stack, result_stack, prev, grammar):
        if prev is None or self in prev.children:
            stack.append(self._sampler)
        else:
            obj_to_check = result_stack.pop()

            def is_acceptable(gamma):
                return bern(
                    (1 / self._alpha_l_u) * (gamma.l_size / (gamma.u_size + 1))
                )

            if is_acceptable(obj_to_check):
                stack.pop()
                result_stack.append(
                    LDerivedClass(obj_to_check.base_class_object))
            else:
                stack.append(self._sampler)


class AliasSampler(BoltzmannSamplerBase):
    """
    Sampler defined by a rule in a decomposition grammar.

    The rule can contain the same alias sampler itself, allowing the
    implementation of recursive decompositions. From a different point of view,
    an alias sampler is a non-terminal symbol in the decomposition grammar.

    Parameters
    ----------
    alias : str
    """

    __slots__ = '_alias', '_grammar', '_referenced_sampler'

    def __init__(self, alias):
        super(AliasSampler, self).__init__()
        self._alias = alias
        self._grammar = None
        self._referenced_sampler = None  # The referenced sampler.
        self.children = None

    def _grammar_not_initialized_error_msg(self):
        return "{}: you have to set the grammar this AliasSampler belongs to" \
               " before using it".format(self._alias)

    @property
    def is_recursive(self):
        """Checks if this AliasSampler belongs to a recursive rule
        (i.e. calls itself directly or indirectly).

        Returns
        -------
        bool
            True iff this sampler corresponds to a recursive rule in the
            grammar.
        """
        return self._grammar.is_recursive_rule(self._alias)

    @property
    def grammar(self):
        """Returns the grammar this alias sampler belongs to.

        Returns
        -------
        grammar: DecompositionGrammar
        """
        return self._grammar

    @grammar.setter
    def grammar(self, g):
        """Sets the grammar of this sampler.

        Parameters
        ----------
        g: DecompositionGrammar
        """
        self._grammar = g

    @property
    def sampled_class(self):
        return self._alias

    @return_precomp
    def eval(self, x, y):
        if self.grammar is None:
            raise BoltzmannFrameworkError(
                self._grammar_not_initialized_error_msg())
        if self.is_recursive:
            return self.oracle.get(self.oracle_query_string(x, y))
        else:
            return self._referenced_sampler.eval(x, y)

    def oracle_query_string(self, x, y):
        if self.grammar is None:
            raise BoltzmannFrameworkError(
                self._grammar_not_initialized_error_msg())
        if self.is_recursive:
            # Evaluations of recursive classes can't be inferred automatically.
            return "{}({},{})".format(self._alias, x, y)
        else:
            return self._referenced_sampler.oracle_query_string(x, y)

    def get_children(self):
        return self._referenced_sampler,

    def accept(self, visitor):
        # Here we let the visitor decide if he wants to recurse further down
        # into the children.
        # (Kind of ugly but needed to avoid infinite recursion.)
        visitor_wants_to_go_on = visitor.visit(self)
        if visitor_wants_to_go_on:
            for child in self.get_children():
                child.accept(visitor)

    def sample_iterative(self, stack, result_stack, prev, grammar):
        if prev is None or self in prev.children:
            stack.append(self._referenced_sampler)
        else:
            stack.pop()
