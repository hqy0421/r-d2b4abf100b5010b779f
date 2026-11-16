from agate.nested import NestedTables


@classmethod
def from_nested_object(cls, obj, schema=None):
    """
    Split a nested Python object into a hierarchy of native tables according
    to an explicit schema.

    This is the schema-aware counterpart of :meth:`.Table.from_object`.
    Instead of flattening nested paths into columns, it returns a
    :class:`.NestedTables` instance whose :attr:`.NestedTables.root` table and
    child tables are ordinary, independently analyzable :class:`.Table`
    objects. See :class:`.NestedTables` and :meth:`.NestedTables.from_object`
    for the schema format, system columns and rebuilding rules.

    :param obj:
        A JSON-like Python object (dicts, lists/tuples, strings, numbers,
        booleans and ``None``).
    :param schema:
        The explicit schema node, or ``None`` to infer strict JSON-native
        types from the data.
    :returns:
        A :class:`.NestedTables` instance.
    """
    return NestedTables.from_object(obj, schema=schema)
