Nested tables
=============

The :class:`.NestedTables` container splits a nested JSON-like object into a
hierarchy of ordinary :class:`.Table` objects according to an explicit schema,
and rebuilds the nested object from the *current* contents of those tables.

Use :meth:`.Table.from_nested_object` (or :meth:`.Table.from_nested_json` for a
JSON document) instead of :meth:`.Table.from_object` when you want to keep
nested objects and arrays as separate, independently analyzable tables rather
than flattening their paths into slash-separated column names.

Schema
------

A schema is a nested structure of plain Python objects:

* a :class:`.DataType` instance describes a scalar field;
* a :class:`dict` maps object field names to child schema nodes;
* a single-item :class:`list` describes an array; its one element is the
  schema for every item in the array.

For example::

    schema = {
        'id': agate.Number(),
        'name': agate.Text(cast_nulls=False),
        'tags': [agate.Text(cast_nulls=False)],
        'address': {
            'street': agate.Text(cast_nulls=False),
            'zip': agate.Text(cast_nulls=False),
        },
        'items': [
            {
                'sku': agate.Text(cast_nulls=False),
                'price': agate.Number(),
            }
        ],
    }

    nested = agate.Table.from_nested_object(data, schema)

If ``schema`` is omitted, it is inferred from the data using strict JSON-native
types: strings are always text (even ``""`` or numeric-looking strings),
booleans stay booleans, and numbers are :class:`.Number` columns. A column that
only ever contains ``null`` becomes a null-preserving :class:`.Text` column.
Conflicting observations, such as a boolean in one row and a number in another,
raise :class:`.NestedValidationError` rather than silently widening the type.

System columns
--------------

Each table is linked to the rest of the hierarchy by reserved system columns:

* ``__id`` -- a unique identifier for every row.
* ``__parent`` -- the ``__id`` of the row's parent row (null for the root).
* ``__index`` -- the position of an element within its parent array. Object
  tables do not have this column.
* ``__present`` -- a JSON object recording, per row, which fields are present
  and whether a container field is an empty container or ``null``. This keeps
  missing fields, explicit ``null``, empty objects and empty arrays distinct.

These column names are reserved and may not be used as data keys.

Working with the tables
-----------------------

Object fields are ordinary columns, so they work with every table method,
including :meth:`.Table.compute`. Array elements are rows, so they can be
filtered with :meth:`.Table.where`, reordered with :meth:`.Table.order_by`, or
added to by constructing a new table. After changing an array's row set, call
:meth:`.NestedTables.reindex` to renumber the ``__index`` column, then pass the
new tables to :meth:`.NestedTables.replace`.

Rebuilding
----------

:meth:`.NestedTables.to_object` and :meth:`.NestedTables.to_json` rebuild the
nested object entirely from the current tables. The original object is never
consulted, so deleted rows stay deleted and added rows are included.

If the relationship keys, ordering or presence states no longer describe a
single consistent hierarchy, the rebuild raises
:class:`.NestedValidationError` listing every problem found. The tables and any
previously rebuilt result are left unchanged.

:class:`decimal.Decimal` values are written as exact JSON numbers; reading a
document with :meth:`.Table.from_nested_json` parses floats as
:class:`~decimal.Decimal` so numeric values round-trip without precision loss.

Reference
---------

.. autoclass:: agate.NestedTables
    :members: from_object, from_json, tables, root, schema, names, replace,
        validate, to_object, to_json, reindex

.. autoclass:: agate.NestedValidationError
    :members:

.. autofunction:: agate.json_dumps
