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

"""
Several combinatorial class builders: abstract, default and dummy.

User defined builders should derive from `CombinatorialClassBuilder`.
Builders are responsable for creating the output of the generic samplers.
"""

import pyboltzmann as pybo

__all__ = ['CombinatorialClassBuilder',
           'DefaultBuilder',
           'DummyBuilder']


class CombinatorialClassBuilder(object):
    """
    Interface for objects that build combinatorial classes.
    # TODO Implement a size checking mechanism?
    """

    def zero_atom(self):
        """Builds a zero-atom.

        Returns
        -------
        CombinatorialClass
        """
        raise NotImplementedError

    def l_atom(self):
        """Builds an l-atom.

        Returns
        -------
        CombinatorialClass
        """
        raise NotImplementedError

    def u_atom(self):
        """Builds a u-atom.

        Returns
        -------
        CombinatorialClass
        """
        raise NotImplementedError

    def product(self, lhs, rhs):
        """Builds an object which corresponds to a combinatorial product of
        the given arguments.

        Parameters
        ----------
        lhs : CombinatorialClass
        rhs : CombinatorialClass

        Returns
        -------
        CombinatorialClass
        """
        raise NotImplementedError

    def set(self, elements):
        """Builds an object which corresponds to a combinatorial set
        construction of the given elements.

        Parameters
        ----------
        elements : list of CombinatorialClass

        Returns
        -------
        CombinatorialClass
        """
        raise NotImplementedError


class DefaultBuilder(CombinatorialClassBuilder):
    """
    Builds the generic class objects.
    """

    def zero_atom(self):
        return pybo.ZeroAtomClass()

    def l_atom(self):
        return pybo.LAtomClass()

    def u_atom(self):
        return pybo.UAtomClass()

    def product(self, lhs, rhs):
        return pybo.ProdClass(lhs, rhs)

    def set(self, elements):
        return pybo.SetClass(elements)


class DummyBuilder(CombinatorialClassBuilder):
    """
    Builds dummy objects.
    """

    def zero_atom(self):
        return pybo.DummyClass()

    def l_atom(self):
        return pybo.DummyClass(l_size=1)

    def u_atom(self):
        return pybo.DummyClass(u_size=1)

    def product(self, lhs, rhs):
        lhs._l_size += rhs._l_size
        lhs._u_size += rhs._u_size
        return lhs

    def set(self, elements):
        l_size = 0
        u_size = 0
        for dummy in elements:
            l_size += dummy.l_size
            u_size += dummy.u_size
        return pybo.DummyClass(l_size, u_size)
