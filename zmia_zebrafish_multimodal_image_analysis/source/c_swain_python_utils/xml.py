#!python3
# xml.py

# Project: c_swain_python_utils
# by Corban Swain 2021

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from xml.etree.ElementTree import Element
import copy

__all__ = ['XML', 'load_xml', 'print_xml']


def load_xml(xml_path):
    tree = ElementTree.parse(xml_path)
    return tree.getroot()


def print_xml(xml_like, **kwargs):
    XML(xml_like).print(**kwargs)


class XML(Element):
    def __init__(self, xml_like: str | Element | XML):
        input_type = type(xml_like)

        self.root: Element
        if input_type is str:
            self.root = load_xml(xml_like)
        elif input_type is Element:
            self.root = xml_like
        elif input_type is self.__class__:
            self.root = copy.copy(xml_like.root)
        else:
            raise TypeError(f'Unexpected input of type {input_type} passed.')

    def print(self, *,
              child_limit=10,
              recursive=True,
              indent_str='   ',
              line_len_limit=80,
              logger=None,
              _level=0):

        log = logger.debug if logger else print

        if _level == 0:
            log((indent_str * _level) + f'{self.root.tag} {self.root.attrib}')

        if child_limit == 0:
            return

        counter = 0

        truncate_str = '...}'
        if line_len_limit:
            truncate_len = line_len_limit - len(truncate_str)

        full_indent_str = indent_str * (_level + 1)

        for child in self.root:
            attrib_str = str(child.attrib)
            if line_len_limit and len(attrib_str) > line_len_limit:
                attrib_str = attrib_str[:truncate_len] + truncate_str

            log(full_indent_str + f'{child.tag} {attrib_str}')

            if child_limit:
                counter += 1

            if recursive and len(child) > 0:
                if child_limit:
                    recur_child_limit = child_limit - counter
                else:
                    recur_child_limit = None

                self.__class__(child).print(
                    child_limit=recur_child_limit,
                    recursive=True,
                    indent_str=indent_str,
                    logger=logger,
                    line_len_limit=line_len_limit,
                    _level=(_level + 1))

                if child_limit:
                    counter += len(child)

            if child_limit and counter >= child_limit:
                break

    def find(self, *args, **kwargs):
        return self.__class__(self.root.find(*args, **kwargs))

