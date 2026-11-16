import json
from collections import OrderedDict
from decimal import Decimal

from agate.nested import NestedTables


@classmethod
def from_nested_json(cls, path, schema=None, encoding='utf-8', **kwargs):
    """
    Create a hierarchy of native tables from a nested JSON document.

    The document is deserialized with floats parsed as
    :class:`decimal.Decimal` and then passed to
    :meth:`.Table.from_nested_object`. See :class:`.NestedTables` and
    :meth:`.NestedTables.from_json` for the schema format and rebuilding rules.

    :param path:
        Filepath or file-like object from which to read JSON data.
    :param schema:
        The explicit schema node, or ``None`` to infer strict JSON-native
        types from the data.
    :param encoding:
        The encoding of the file when ``path`` is a filepath.
    :code:`kwargs` are passed through to :func:`json.load`.
    :returns:
        A :class:`.NestedTables` instance.
    """
    close = False

    try:
        if hasattr(path, 'read'):
            obj = json.load(path, object_pairs_hook=OrderedDict, parse_float=Decimal, **kwargs)
        else:
            f = open(path, encoding=encoding)
            close = True
            obj = json.load(f, object_pairs_hook=OrderedDict, parse_float=Decimal, **kwargs)
    finally:
        if close:
            f.close()

    return NestedTables.from_object(obj, schema=schema)
