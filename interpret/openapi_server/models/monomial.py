from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from openapi_server.models.base_model import Model
from openapi_server import util


class Monomial(Model):
    """NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).

    Do not edit the class manually.
    """

    def __init__(self, variable=None, coefficient=None, power=None):  # noqa: E501
        """Monomial - a model defined in OpenAPI

        :param variable: The variable of this Monomial.  # noqa: E501
        :type variable: str
        :param coefficient: The coefficient of this Monomial.  # noqa: E501
        :type coefficient: int
        :param power: The power of this Monomial.  # noqa: E501
        :type power: int
        """
        self.openapi_types = {
            'variable': str,
            'coefficient': int,
            'power': int
        }

        self.attribute_map = {
            'variable': 'variable',
            'coefficient': 'coefficient',
            'power': 'power'
        }

        self._variable = variable
        self._coefficient = coefficient
        self._power = power

    @classmethod
    def from_dict(cls, dikt) -> 'Monomial':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The Monomial of this Monomial.  # noqa: E501
        :rtype: Monomial
        """
        return util.deserialize_model(dikt, cls)

    @property
    def variable(self) -> str:
        """Gets the variable of this Monomial.


        :return: The variable of this Monomial.
        :rtype: str
        """
        return self._variable

    @variable.setter
    def variable(self, variable: str):
        """Sets the variable of this Monomial.


        :param variable: The variable of this Monomial.
        :type variable: str
        """
        if variable is None:
            raise ValueError("Invalid value for `variable`, must not be `None`")  # noqa: E501
        if variable is not None and len(variable) < 1:
            raise ValueError("Invalid value for `variable`, length must be greater than or equal to `1`")  # noqa: E501

        self._variable = variable

    @property
    def coefficient(self) -> int:
        """Gets the coefficient of this Monomial.


        :return: The coefficient of this Monomial.
        :rtype: int
        """
        return self._coefficient

    @coefficient.setter
    def coefficient(self, coefficient: int):
        """Sets the coefficient of this Monomial.


        :param coefficient: The coefficient of this Monomial.
        :type coefficient: int
        """
        if coefficient is not None and coefficient < 1:  # noqa: E501
            raise ValueError("Invalid value for `coefficient`, must be a value greater than or equal to `1`")  # noqa: E501

        self._coefficient = coefficient

    @property
    def power(self) -> int:
        """Gets the power of this Monomial.


        :return: The power of this Monomial.
        :rtype: int
        """
        return self._power

    @power.setter
    def power(self, power: int):
        """Sets the power of this Monomial.


        :param power: The power of this Monomial.
        :type power: int
        """
        if power is not None and power < 1:  # noqa: E501
            raise ValueError("Invalid value for `power`, must be a value greater than or equal to `1`")  # noqa: E501

        self._power = power
