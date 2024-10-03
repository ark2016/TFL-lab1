from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from openapi_server.models.base_model import Model
from openapi_server import util


class Letter(Model):
    """NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).

    Do not edit the class manually.
    """

    def __init__(self, name=None, is_variable=None):  # noqa: E501
        """Letter - a model defined in OpenAPI

        :param name: The name of this Letter.  # noqa: E501
        :type name: str
        :param is_variable: The is_variable of this Letter.  # noqa: E501
        :type is_variable: bool
        """
        self.openapi_types = {
            'name': str,
            'is_variable': bool
        }

        self.attribute_map = {
            'name': 'name',
            'is_variable': 'isVariable'
        }

        self._name = name
        self._is_variable = is_variable

    @classmethod
    def from_dict(cls, dikt) -> 'Letter':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The Letter of this Letter.  # noqa: E501
        :rtype: Letter
        """
        return util.deserialize_model(dikt, cls)

    @property
    def name(self) -> str:
        """Gets the name of this Letter.


        :return: The name of this Letter.
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this Letter.


        :param name: The name of this Letter.
        :type name: str
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")  # noqa: E501
        if name is not None and len(name) < 1:
            raise ValueError("Invalid value for `name`, length must be greater than or equal to `1`")  # noqa: E501

        self._name = name

    @property
    def is_variable(self) -> bool:
        """Gets the is_variable of this Letter.


        :return: The is_variable of this Letter.
        :rtype: bool
        """
        return self._is_variable

    @is_variable.setter
    def is_variable(self, is_variable: bool):
        """Sets the is_variable of this Letter.


        :param is_variable: The is_variable of this Letter.
        :type is_variable: bool
        """
        if is_variable is None:
            raise ValueError("Invalid value for `is_variable`, must not be `None`")  # noqa: E501

        self._is_variable = is_variable