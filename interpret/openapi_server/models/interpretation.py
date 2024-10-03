from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from openapi_server.models.base_model import Model
from openapi_server.models.monomial import Monomial
from openapi_server import util

from openapi_server.models.monomial import Monomial  # noqa: E501


class Interpretation(Model):
    """NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).

    Do not edit the class manually.
    """

    def __init__(self, name=None, args=None, monomials=None):  # noqa: E501
        """Interpretation - a model defined in OpenAPI

        :param name: The name of this Interpretation.  # noqa: E501
        :type name: str
        :param args: The args of this Interpretation.  # noqa: E501
        :type args: List[str]
        :param monomials: The monomials of this Interpretation.  # noqa: E501
        :type monomials: List[Monomial]
        """
        self.openapi_types = {
            'name': str,
            'args': List[str],
            'monomials': List[Monomial]
        }

        self.attribute_map = {
            'name': 'name',
            'args': 'args',
            'monomials': 'monomials'
        }

        self._name = name
        self._args = args
        self._monomials = monomials

    @classmethod
    def from_dict(cls, dikt) -> 'Interpretation':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The Interpretation of this Interpretation.  # noqa: E501
        :rtype: Interpretation
        """
        return util.deserialize_model(dikt, cls)

    @property
    def name(self) -> str:
        """Gets the name of this Interpretation.


        :return: The name of this Interpretation.
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this Interpretation.


        :param name: The name of this Interpretation.
        :type name: str
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")  # noqa: E501
        if name is not None and len(name) < 1:
            raise ValueError("Invalid value for `name`, length must be greater than or equal to `1`")  # noqa: E501

        self._name = name

    @property
    def args(self) -> List[str]:
        """Gets the args of this Interpretation.


        :return: The args of this Interpretation.
        :rtype: List[str]
        """
        return self._args

    @args.setter
    def args(self, args: List[str]):
        """Sets the args of this Interpretation.


        :param args: The args of this Interpretation.
        :type args: List[str]
        """
        if args is None:
            raise ValueError("Invalid value for `args`, must not be `None`")  # noqa: E501

        self._args = args

    @property
    def monomials(self) -> List[Monomial]:
        """Gets the monomials of this Interpretation.


        :return: The monomials of this Interpretation.
        :rtype: List[Monomial]
        """
        return self._monomials

    @monomials.setter
    def monomials(self, monomials: List[Monomial]):
        """Sets the monomials of this Interpretation.


        :param monomials: The monomials of this Interpretation.
        :type monomials: List[Monomial]
        """
        if monomials is None:
            raise ValueError("Invalid value for `monomials`, must not be `None`")  # noqa: E501

        self._monomials = monomials