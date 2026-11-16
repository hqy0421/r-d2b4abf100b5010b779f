"""
Split nested JSON-like Python objects into a hierarchy of native
:class:`.Table` objects according to an explicit schema, and rebuild the
nested object from the *current* contents of those tables.

Unlike :meth:`.Table.from_object`, which flattens nested paths into slash-
and dot-separated column names, this module keeps every object and array as
its own table. Object fields remain normal columns that can be used with
:meth:`.Table.compute`, while array elements become rows that can be filtered
out, added or reordered with the usual table methods.

The tables are linked by explicit relationship columns:

* ``__id`` -- a unique identifier for every row (:class:`.Text`).
* ``__parent`` -- the ``__id`` of the row's parent row (:class:`.Text`), or
  null for root rows.
* ``__index`` -- the position of an element within its parent array
  (:class:`.Number`). Object tables do not have this column.
* ``__present`` -- a JSON object recording, for each row, which fields are
  present and whether a container field is an empty container or ``null``
  (:class:`.Text`). This is what keeps missing fields, ``null``, empty
  objects and empty arrays distinct.

Rebuilding never consults the original object or a stored template. It reads
the tables as they currently are and fails with a
:class:`NestedValidationError` if the relationship keys, ordering or presence
states no longer describe a consistent hierarchy. In that case the tables and
any previously rebuilt result are left untouched.

Schema nodes are plain Python objects:

* a :class:`.DataType` instance describes a scalar field;
* a :class:`dict` maps object field names to child schema nodes;
* a single-item :class:`list` describes an array whose one element is the
  item schema.

For example, this schema::

    {
        'id': agate.Number(),
        'name': agate.Text(cast_nulls=False),
        'tags': [agate.Text(cast_nulls=False)],
        'address': {
            'street': agate.Text(cast_nulls=False)
        }
    }

describes an object with scalar fields ``id`` and ``name``, a scalar array
``tags`` and a nested object ``address``.
"""

import datetime
import json
import os
from collections import OrderedDict
from decimal import Decimal

from agate.data_types import Boolean, DataType, Number, Text

#: Column holding the unique identifier of every row.
ID = '__id'
#: Column holding the identifier of the row's parent row.
PARENT = '__parent'
#: Column holding the position of an array element.
INDEX = '__index'
#: Column holding the per-row presence/state map, serialized as JSON.
PRESENT = '__present'
#: Column holding the value of a scalar root or a scalar array element.
VALUE = '__value'
#: Synthetic key recording the state of a root array container row.
ROOT_STATE = '$'

#: Column names reserved by the nested-table machinery.
RESERVED_COLUMNS = frozenset((ID, PARENT, INDEX, PRESENT, VALUE))

_SYSTEM_TYPES = {
    ID: Text(cast_nulls=False),
    PARENT: Text(cast_nulls=False),
    INDEX: Number(),
    PRESENT: Text(cast_nulls=False),
}

_NULL = 'null'
_EMPTY = 'empty'
_SCALAR = 'scalar'
_OBJECT = 'object'
_ARRAY = 'array'


class NestedValidationError(ValueError):
    """
    Raised when nested tables can not be split or rebuilt consistently.

    Every inconsistency found during an operation is listed in
    :attr:`messages`, so a single failure reports all of the relationship,
    ordering, state or schema problems rather than just the first one.
    """
    def __init__(self, messages):
        self.messages = tuple(messages)

        super().__init__('; '.join(self.messages))


def _render_path(path):
    if not path:
        return '$'

    return '$' + ''.join(part if part.startswith('[') else '.' + part for part in path)


def _native_data_type(value):
    """
    Infer a data type from a native JSON value without coercing strings or
    booleans into numbers and without treating strings such as ``""`` or
    ``"null"`` as nulls. Inferred text columns always keep their literal
    string value, since JSON itself has no concept of null-like text tokens.
    """
    if type(value) is bool:
        return Boolean()
    if isinstance(value, (int, float, Decimal)):
        return Number()
    if isinstance(value, str):
        return Text(cast_nulls=False)

    return None


def _format_decimal(value):
    """
    Format a :class:`decimal.Decimal` as a JSON number, preserving trailing
    zeroes and expanding scientific notation. For example, ``Decimal('1.50')``
    renders as ``1.50`` and ``Decimal('1E+2')`` as ``100``.
    """
    return format(value, 'f')


def json_dumps(obj, indent=None, ensure_ascii=False):
    """
    Serialize a nested Python object to JSON while emitting
    :class:`decimal.Decimal` values as exact JSON numbers instead of floats.

    String escaping is delegated to :func:`json.dumps`; only structural
    rendering is custom, so the output is valid JSON that round-trips through
    ``json.loads(..., parse_float=Decimal)`` without precision loss.
    """
    def escape(text):
        return json.dumps(text, ensure_ascii=ensure_ascii)

    def render(value, level):
        if value is None:
            return 'null'
        if value is True:
            return 'true'
        if value is False:
            return 'false'
        if isinstance(value, Decimal):
            return _format_decimal(value)
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return repr(value)
        if isinstance(value, str):
            return escape(value)
        if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
            return escape(value.isoformat())
        if isinstance(value, datetime.timedelta):
            return escape(str(value))
        if isinstance(value, dict):
            if not value:
                return '{}'

            if indent is None:
                parts = [
                    escape(str(key)) + ': ' + render(item, level + 1)
                    for key, item in value.items()
                ]

                return '{' + ', '.join(parts) + '}'

            inner_pad = '\n' + (' ' * (indent * (level + 1)))
            outer_pad = '\n' + (' ' * (indent * level))
            parts = [
                inner_pad + escape(str(key)) + ': ' + render(item, level + 1)
                for key, item in value.items()
            ]

            return '{' + ','.join(parts) + outer_pad + '}'
        if isinstance(value, (list, tuple)):
            if not value:
                return '[]'

            if indent is None:
                return '[' + ', '.join(render(item, level + 1) for item in value) + ']'

            inner_pad = '\n' + (' ' * (indent * (level + 1)))
            outer_pad = '\n' + (' ' * (indent * level))

            return '[' + ','.join(inner_pad + render(item, level + 1) for item in value) + outer_pad + ']'

        raise TypeError('Object of type %s is not JSON serializable' % type(value).__name__)

    return render(obj, 0)


# ---------------------------------------------------------------------------
# Schema construction
# ---------------------------------------------------------------------------

def _infer_node(value, path, errors):
    """
    Build an internal schema node by inferring strict JSON-native types from a
    single value.
    """
    if value is None:
        return {'kind': 'scalar', 'data_type': None}
    if isinstance(value, dict):
        fields = OrderedDict()

        for key, item in value.items():
            if not isinstance(key, str):
                errors.append('At %s: object keys must be strings, got %s.'
                              % (_render_path(path), type(key).__name__))
                continue
            if key in RESERVED_COLUMNS:
                errors.append('At %s: key %r is reserved for nested-table system columns.'
                              % (_render_path(path), key))
                continue

            fields[key] = _infer_node(item, path + (key,), errors)

        return {'kind': 'object', 'fields': fields}
    if isinstance(value, (list, tuple)):
        item_node = {'kind': 'scalar', 'data_type': None}

        for i, item in enumerate(value):
            candidate = _infer_node(item, path + ('[%d]' % i,), errors)
            item_node = _merge_node(item_node, candidate, path, errors)

        return {'kind': 'array', 'item': item_node}

    data_type = _native_data_type(value)

    if data_type is None:
        errors.append('At %s: value %r of type %s is not a supported JSON value.'
                      % (_render_path(path), value, type(value).__name__))
        data_type = Text(cast_nulls=False)

    return {'kind': 'scalar', 'data_type': data_type}


def _merge_node(left, right, path, errors):
    """
    Merge two inferred nodes describing the same position across rows.
    Conflicting scalar types (including boolean versus number) are an explicit
    failure instead of a lossy widening.
    """
    # A node inferred solely from null values has no data type yet and adopts
    # the shape of any later, non-null observation.
    if left['kind'] == 'scalar' and left['data_type'] is None:
        return right
    if right['kind'] == 'scalar' and right['data_type'] is None:
        return left

    if left['kind'] != right['kind']:
        errors.append('At %s: conflicting value kinds %s and %s. Specify an explicit schema to resolve it.'
                      % (_render_path(path), left['kind'], right['kind']))
        return left

    if left['kind'] == 'scalar':
        if type(left['data_type']) is not type(right['data_type']):
            errors.append(
                'At %s: conflicting scalar types %s and %s. Specify an explicit schema to resolve it.'
                % (
                    _render_path(path),
                    type(left['data_type']).__name__,
                    type(right['data_type']).__name__,
                )
            )

        return left

    if left['kind'] == 'object':
        fields = OrderedDict(left['fields'])

        for key, node in right['fields'].items():
            if key in fields:
                fields[key] = _merge_node(fields[key], node, path + (key,), errors)
            else:
                fields[key] = node

        return {'kind': 'object', 'fields': fields}

    return {
        'kind': 'array',
        'item': _merge_node(left['item'], right['item'], path + ('[]',), errors),
    }


def _normalize_schema(node, path, errors):
    """
    Convert a user-supplied schema into an internal schema node.
    """
    if isinstance(node, DataType):
        return {'kind': 'scalar', 'data_type': node}
    if isinstance(node, dict):
        fields = OrderedDict()

        for key, child in node.items():
            if not isinstance(key, str):
                errors.append('At %s: schema object keys must be strings, got %s.'
                              % (_render_path(path), type(key).__name__))
                continue
            if key in RESERVED_COLUMNS:
                errors.append('At %s: schema key %r is reserved for nested-table system columns.'
                              % (_render_path(path), key))
                continue

            fields[key] = _normalize_schema(child, path + (key,), errors)

        return {'kind': 'object', 'fields': fields}
    if isinstance(node, list):
        if len(node) != 1:
            errors.append('At %s: an array schema must contain exactly one item schema, got %d.'
                          % (_render_path(path), len(node)))
            return {'kind': 'array', 'item': {'kind': 'scalar', 'data_type': Text(cast_nulls=False)}}

        return {'kind': 'array', 'item': _normalize_schema(node[0], path + ('[]',), errors)}

    errors.append(
        'At %s: schema node must be a DataType instance, an object dict or a single-item list, got %s.'
        % (_render_path(path), type(node).__name__)
    )
    return {'kind': 'scalar', 'data_type': Text(cast_nulls=False)}


def _finalize_types(node):
    """
    Resolve any scalar node that never observed a non-null value to strict
    text, so an all-null or never-populated column still has a concrete,
    null-preserving data type instead of falling back to type testing.
    """
    kind = node['kind']

    if kind == 'scalar':
        if node['data_type'] is None:
            node['data_type'] = Text(cast_nulls=False)
    elif kind == 'object':
        for child in node['fields'].values():
            _finalize_types(child)
    else:
        _finalize_types(node['item'])


def _validate_value_against_schema(value, node, path, errors):
    """
    Pre-flight kind checks for explicit schemas. Scalar casting is validated
    again while building so cast errors carry full location context.
    """
    kind = node['kind']

    if kind == 'scalar':
        if value is None:
            return
        if not isinstance(value, (bool, int, float, Decimal, str)):
            errors.append('At %s: value %r of type %s is not a supported JSON scalar.'
                          % (_render_path(path), value, type(value).__name__))
        elif node['data_type'] is not None:
            try:
                node['data_type'].cast(value)
            except Exception as e:
                errors.append('At %s: %s' % (_render_path(path), e))
        return

    if value is None:
        return

    if kind == 'object':
        if not isinstance(value, dict):
            errors.append('At %s: expected an object, got %s.' % (_render_path(path), type(value).__name__))
            return

        extra = set(value.keys()) - set(node['fields'].keys())

        if extra:
            errors.append('At %s: object contains fields not in the schema: %s.'
                          % (_render_path(path), ', '.join(sorted(extra))))

        for key, child in node['fields'].items():
            if key in value:
                _validate_value_against_schema(value[key], child, path + (key,), errors)

        return

    if not isinstance(value, (list, tuple)):
        errors.append('At %s: expected an array, got %s.' % (_render_path(path), type(value).__name__))
        return

    for i, item in enumerate(value):
        _validate_value_against_schema(item, node['item'], path + ('[%d]' % i,), errors)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

class _Builder:
    """
    Walks a nested value once, collecting rows for every object and array node.

    Tables are pre-registered from the schema before walking, so even empty
    objects and empty arrays get a native table that can be analyzed or have
    rows added later. Tables are only constructed at the end, which means a
    validation failure produces no tables at all.
    """
    def __init__(self, root_node, errors):
        self._root_node = root_node
        self._errors = errors
        self._rows = OrderedDict()
        self._columns = OrderedDict()
        self._counter = 0
        self.path_names = OrderedDict()
        self._used_names = set()

    def _new_id(self):
        self._counter += 1

        return 'n%d' % self._counter

    def _register(self, path, column_names, column_types):
        if path in self._rows:
            return

        self._rows[path] = []
        self._columns[path] = (tuple(column_names), tuple(column_types))

        name = 'root' if not path else '/'.join(str(part) for part in path)
        candidate = name
        suffix = 2

        while candidate in self._used_names:
            candidate = '%s_%d' % (name, suffix)
            suffix += 1

        self._used_names.add(candidate)
        self.path_names[path] = candidate

    def _object_columns(self, node):
        names = [ID, PARENT, PRESENT]
        types = [_SYSTEM_TYPES[ID], _SYSTEM_TYPES[PARENT], _SYSTEM_TYPES[PRESENT]]

        for key, child in node['fields'].items():
            if child['kind'] == 'scalar':
                names.append(key)
                types.append(child['data_type'] or Text(cast_nulls=False))

        return names, types

    def _array_columns(self, path, node):
        item = node['item']
        names = [ID, PARENT, INDEX, PRESENT]
        types = [_SYSTEM_TYPES[ID], _SYSTEM_TYPES[PARENT], _SYSTEM_TYPES[INDEX], _SYSTEM_TYPES[PRESENT]]

        if item['kind'] == 'scalar':
            value_column = VALUE if not path else path[-1]
            names.append(value_column)
            types.append(item['data_type'] or Text(cast_nulls=False))
        elif item['kind'] == 'object':
            for key, child in item['fields'].items():
                if child['kind'] == 'scalar':
                    names.append(key)
                    types.append(child['data_type'] or Text(cast_nulls=False))

        return names, types

    def _pre_register(self, node, path):
        """Create an (empty) table slot for every node in the schema."""
        kind = node['kind']

        if kind == 'scalar':
            if path == ():
                self._register(
                    path,
                    (ID, PARENT, PRESENT, VALUE),
                    (_SYSTEM_TYPES[ID], _SYSTEM_TYPES[PARENT], _SYSTEM_TYPES[PRESENT],
                     node['data_type'] or Text(cast_nulls=False)),
                )
            return

        if kind == 'object':
            self._register(path, *self._object_columns(node))

            for key, child in node['fields'].items():
                self._pre_register(child, path + (key,))
        else:
            self._register(path, *self._array_columns(path, node))

            # An array's items live in the array table itself; only the item's
            # own nested fields introduce additional tables.
            item = node['item']

            if item['kind'] == 'object':
                for key, child in item['fields'].items():
                    self._pre_register(child, path + (key,))

    def build(self, value):
        self._pre_register(self._root_node, ())

        kind = self._root_node['kind']

        if kind == 'scalar':
            present = OrderedDict()

            if value is None:
                present[VALUE] = _NULL
                row_value = None
            else:
                present[VALUE] = _SCALAR
                row_value = self._cast_scalar(value, self._root_node, ())

            self._rows[()].append(('root', None, json.dumps(present, ensure_ascii=False), row_value))
        elif kind == 'object':
            present, scalars = self._walk_object(value, self._root_node, (), 'root', None)
            self._rows[()].append(self._object_row(self._root_node, 'root', None, present, scalars))
        else:
            self._walk_root_array(value, self._root_node, ())

        return self._make_tables()

    def _walk_root_array(self, value, node, path):
        if value is None:
            self._rows[path].append(('root', None, None, json.dumps({ROOT_STATE: _NULL})))
            return
        if not isinstance(value, (list, tuple)):
            self._errors.append('At %s: expected an array, got %s.' % (_render_path(path), type(value).__name__))
            return

        if not value:
            self._rows[path].append(('root', None, None, json.dumps({ROOT_STATE: _EMPTY})))
            return

        self._rows[path].append(('root', None, None, json.dumps({})))

        for index, item in enumerate(value):
            self._walk_array_item(item, node['item'], path, 'root', index)

    def _walk_object(self, value, node, path, row_id, parent_id):
        """
        Walk a non-null object value, creating rows for nested container
        fields. Returns a ``(present, scalars)`` pair: ``present`` maps each
        present field to its state, and ``scalars`` holds the cast values for
        present scalar fields (including explicit nulls).
        """
        if not isinstance(value, dict):
            self._errors.append('At %s: expected an object, got %s.' % (_render_path(path), type(value).__name__))
            return OrderedDict(), OrderedDict()

        present = OrderedDict()
        scalars = OrderedDict()

        for key, child in node['fields'].items():
            if key not in value:
                # A missing field is simply absent from the presence map.
                continue

            item = value[key]
            child_path = path + (key,)
            kind = child['kind']

            if kind == 'scalar':
                if item is None:
                    present[key] = _NULL
                    scalars[key] = None
                else:
                    present[key] = _SCALAR
                    scalars[key] = self._cast_scalar(item, child, child_path)

                continue

            if item is None:
                present[key] = _NULL
                continue

            if kind == 'object':
                if not item:
                    # An empty object owns no child row (mirroring an empty
                    # array), but its pre-registered table still exists and can
                    # be analyzed or populated later together with a state
                    # change to "object".
                    present[key] = _EMPTY
                else:
                    present[key] = _OBJECT
                    child_id = self._new_id()
                    child_present, child_scalars = self._walk_object(
                        item, child, child_path, child_id, row_id
                    )
                    self._rows[child_path].append(
                        self._object_row(child, child_id, row_id, child_present, child_scalars)
                    )
            else:
                if not item:
                    present[key] = _EMPTY
                else:
                    present[key] = _ARRAY
                    self._walk_array_items(item, child, child_path, row_id)

        extra = set(value.keys()) - set(node['fields'].keys())

        if extra:
            self._errors.append('At %s: object contains fields not in the schema: %s.'
                                % (_render_path(path), ', '.join(sorted(extra))))

        return present, scalars

    def _walk_array_items(self, value, node, path, parent_id):
        if not isinstance(value, (list, tuple)):
            self._errors.append('At %s: expected an array, got %s.' % (_render_path(path), type(value).__name__))
            return

        for index, item in enumerate(value):
            self._walk_array_item(item, node['item'], path, parent_id, index)

    def _walk_array_item(self, item, item_node, path, parent_id, index):
        row_id = self._new_id()
        kind = item_node['kind']

        if kind == 'scalar':
            cast_value = None if item is None else self._cast_scalar(
                item, item_node, path + ('[%d]' % index,)
            )
            self._rows[path].append(
                (row_id, parent_id, Decimal(index), json.dumps({}), cast_value)
            )
            return

        if kind == 'array':
            self._errors.append(
                'At %s: nested arrays are not supported; flatten the schema or preprocess the data.'
                % _render_path(path + ('[%d]' % index,))
            )
            return

        if item is None:
            self._errors.append('At %s: array element is null but the item schema is an object.'
                                % _render_path(path + ('[%d]' % index,)))
            return
        if not isinstance(item, dict):
            self._errors.append('At %s: expected an object array element, got %s.'
                                % (_render_path(path + ('[%d]' % index,)), type(item).__name__))
            return

        child_present, child_scalars = self._walk_object(item, item_node, path, row_id, parent_id)
        row = [row_id, parent_id, Decimal(index), json.dumps(child_present, ensure_ascii=False)]

        for key, child in item_node['fields'].items():
            if child['kind'] != 'scalar':
                continue

            row.append(child_scalars.get(key))

        self._rows[path].append(tuple(row))

    def _cast_scalar(self, value, node, path):
        data_type = node['data_type']

        if data_type is None:
            inferred = _native_data_type(value)

            if inferred is None:
                self._errors.append('At %s: value %r of type %s is not a supported JSON value.'
                                    % (_render_path(path), value, type(value).__name__))
                return value

            node['data_type'] = inferred
            data_type = inferred

        if type(value) is bool and isinstance(data_type, Number) and not isinstance(data_type, Boolean):
            self._errors.append('At %s: boolean value %r is not a number; use a Boolean field or quote the value.'
                                % (_render_path(path), value))
            return value

        try:
            return data_type.cast(value)
        except Exception as e:
            self._errors.append('At %s: %s' % (_render_path(path), e))

            return value

    def _object_row(self, node, row_id, parent_id, present, scalars):
        row = [row_id, parent_id, json.dumps(present, ensure_ascii=False)]

        for key, child in node['fields'].items():
            if child['kind'] == 'scalar':
                row.append(scalars.get(key))

        return tuple(row)

    def _make_tables(self):
        from agate.table import Table

        tables = OrderedDict()

        for path, rows in self._rows.items():
            column_names, column_types = self._columns[path]
            name = self.path_names[path]

            try:
                tables[name] = Table(
                    rows, column_names=list(column_names), column_types=list(column_types)
                )
            except Exception as e:
                self._errors.append('Could not build table for %s: %s' % (_render_path(path), e))

        return tables


# ---------------------------------------------------------------------------
# Public container
# ---------------------------------------------------------------------------

class NestedTables:
    """
    A hierarchy of native :class:`.Table` objects representing a nested
    JSON-like object, together with the schema describing their relationship.

    Instances are immutable entry points: filtering, computing, reordering or
    adding rows all produce new :class:`.Table` objects. Pass the modified
    tables back to :meth:`replace` and then call :meth:`to_object` or
    :meth:`to_json` to rebuild the nested object. The rebuild reads the
    current tables and raises :class:`NestedValidationError` on any
    inconsistency without modifying the tables.
    """
    def __init__(self, tables, root_node, path_names):
        self._tables = OrderedDict(tables)
        self._root_node = root_node
        self._path_names = OrderedDict(path_names)

    @classmethod
    def from_object(cls, obj, schema=None):
        """
        Split a nested Python object into parent/child tables.

        :param obj:
            A JSON-like Python object (dicts, lists/tuples, strings, numbers,
            booleans and ``None``). Numbers may be :class:`decimal.Decimal`.
        :param schema:
            An explicit schema node. Scalar fields are :class:`.DataType`
            instances, object fields are dicts mapping field names to child
            schemas, and array fields are single-item lists containing the
            item schema. If ``None``, the schema is inferred from the data
            using strict JSON-native types (strings stay text, booleans stay
            booleans, numbers stay numbers).
        :raises NestedValidationError:
            If the schema is invalid, the data does not match the schema, or
            inferred types conflict.
        """
        errors = []

        if schema is None:
            root_node = _infer_node(obj, (), errors)
        else:
            root_node = _normalize_schema(schema, (), errors)

            if not errors:
                _validate_value_against_schema(obj, root_node, (), errors)

        if errors:
            raise NestedValidationError(errors)

        # Give every scalar column a concrete, null-preserving type before any
        # table is built, so inference never falls back onto TypeTester.
        _finalize_types(root_node)

        builder = _Builder(root_node, errors)
        tables = builder.build(obj)

        if errors:
            raise NestedValidationError(errors)

        return cls(tables, root_node, builder.path_names)

    @classmethod
    def from_json(cls, path, schema=None, encoding='utf-8', **kwargs):
        """
        Read a JSON document and split it into parent/child tables.

        Floats are parsed as :class:`decimal.Decimal` so every numeric value
        round-trips exactly. See :meth:`from_object` for the schema rules.
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

        return cls.from_object(obj, schema=schema)

    @property
    def tables(self):
        """
        An :class:`collections.OrderedDict` of table names to native
        :class:`.Table` instances, including the ``root`` table.
        """
        return self._tables

    @property
    def root(self):
        """
        The root :class:`.Table`.
        """
        return self._tables[self._path_names[()]]

    @property
    def schema(self):
        """
        The normalized internal schema node for the hierarchy.
        """
        return self._root_node

    def __getitem__(self, name):
        return self._tables[name]

    def __contains__(self, name):
        return name in self._tables

    def names(self):
        """
        Return the names of all tables in hierarchy order.
        """
        return tuple(self._tables.keys())

    def replace(self, tables):
        """
        Return a new :class:`NestedTables` using the given tables (a mapping of
        table name to :class:`.Table`). The current instance is not modified.

        The replacement is not validated immediately; call :meth:`validate`,
        :meth:`to_object` or :meth:`to_json` to check consistency.
        """
        tables = OrderedDict(tables)
        missing = set(self._tables) - set(tables)

        if missing:
            raise NestedValidationError(['Missing required table(s): %s.' % ', '.join(sorted(missing))])

        unknown = set(tables) - set(self._tables)

        if unknown:
            raise NestedValidationError(['Unknown table(s): %s.' % ', '.join(sorted(unknown))])

        return NestedTables(tables, self._root_node, self._path_names)

    def validate(self):
        """
        Validate that every table still describes a single consistent
        hierarchy. Returns ``None`` on success and raises
        :class:`NestedValidationError`, listing all problems, otherwise.
        """
        self._rebuild()

    def to_object(self):
        """
        Rebuild and return the nested Python object from the current tables.

        The original object is never read: every value and container comes
        from the current table rows. On failure, :class:`NestedValidationError`
        is raised and nothing is returned.
        """
        return self._rebuild()

    def to_json(self, path, indent=None, ensure_ascii=False):
        """
        Rebuild the nested object from the current tables and write it as
        JSON to a filepath or file-like object.

        :class:`decimal.Decimal` values are written as exact JSON numbers.
        """
        obj = self.to_object()
        text = json_dumps(obj, indent=indent, ensure_ascii=ensure_ascii)
        close = False

        try:
            if hasattr(path, 'write'):
                f = path
            else:
                directory = os.path.dirname(path)

                if directory and not os.path.exists(directory):
                    os.makedirs(directory)

                f = open(path, 'w', encoding='utf-8')
                close = True

            f.write(text)
        finally:
            if close:
                f.close()

    def reindex(self, table):
        """
        Return a copy of an array table in which the ``__index`` column has
        been renumbered ``0..n-1`` in current physical row order.

        Accepts either a table name or a (possibly filtered or reordered)
        :class:`.Table` belonging to one of this hierarchy's array tables.
        Use this after :meth:`.Table.order_by`, :meth:`.Table.where` or adding
        rows so that array ordering matches the physical row order. The
        original table is not modified.
        """
        from agate.table import Table

        label = table

        if isinstance(table, str):
            table = self._tables[table]
        else:
            label = 'array table'

        if INDEX not in table.column_names:
            raise NestedValidationError(
                ['%r has no %s column; only array tables can be reindexed.' % (label, INDEX)]
            )

        index_pos = table.column_names.index(INDEX)
        rows = []

        for i, row in enumerate(table.rows):
            values = list(row.values())
            values[index_pos] = Decimal(i)
            rows.append(tuple(values))

        return Table(
            rows, column_names=list(table.column_names), column_types=list(table.column_types)
        )

    def _rebuild(self):
        validator = _Rebuilder(self._tables, self._path_names)
        result = validator.rebuild_root(self._root_node)

        if validator.errors:
            raise NestedValidationError(validator.errors)

        return result


# ---------------------------------------------------------------------------
# Rebuilding
# ---------------------------------------------------------------------------

class _Rebuilder:
    """
    Rebuilds a nested object entirely from current table rows, collecting every
    relationship, ordering and presence-state inconsistency.
    """
    def __init__(self, tables, path_names):
        self._tables = tables
        self._path_names = path_names
        self.errors = []
        self._prepared = {}

    def _table_name(self, path):
        return self._path_names[path]

    def _table(self, path):
        return self._tables[self._table_name(path)]

    def _prepare(self, path):
        if path in self._prepared:
            return self._prepared[path]

        table = self._table(path)
        by_id = OrderedDict()
        children = OrderedDict()

        self._check_system_columns(path, table)

        for row in table.rows:
            if ID not in table.column_names:
                continue

            row_id = row[ID]

            if row_id in by_id:
                self.errors.append('In table %r: duplicate %s %r.' % (self._table_name(path), ID, row_id))
                continue

            by_id[row_id] = row
            parent = row[PARENT] if PARENT in table.column_names else None
            children.setdefault(parent, []).append(row)

        self._prepared[path] = (table, by_id, children)

        return self._prepared[path]

    def _check_system_columns(self, path, table):
        names = table.column_names
        types = dict(zip(names, table.column_types))

        for column in (ID, PARENT, PRESENT):
            if column not in names:
                self.errors.append('In table %r: missing required system column %r.'
                                   % (self._table_name(path), column))
            elif not isinstance(types[column], type(_SYSTEM_TYPES[column])):
                self.errors.append('In table %r: system column %r must have type %s.'
                                   % (self._table_name(path), column,
                                      type(_SYSTEM_TYPES[column]).__name__))

        if INDEX in names and not isinstance(types[INDEX], Number):
            self.errors.append('In table %r: system column %r must have type Number.'
                               % (self._table_name(path), INDEX))

    def _read_present(self, path, row):
        if PRESENT not in row.keys():
            return {}

        raw = row[PRESENT]

        if not isinstance(raw, str):
            self.errors.append('In table %r row %r: %s must be a JSON-encoded string.'
                               % (self._table_name(path), row[ID], PRESENT))
            return {}

        try:
            present = json.loads(raw)
        except (ValueError, TypeError):
            self.errors.append('In table %r row %r: %s is not valid JSON (%r).'
                               % (self._table_name(path), row[ID], PRESENT, raw))
            return {}

        if not isinstance(present, dict):
            self.errors.append('In table %r row %r: %s must encode a JSON object.'
                               % (self._table_name(path), row[ID], PRESENT))
            return {}

        return present

    def rebuild_root(self, node):
        path = ()
        table, by_id, children = self._prepare(path)

        if node['kind'] == 'scalar':
            if len(by_id) != 1:
                self.errors.append('Root scalar table must contain exactly one row, got %d.' % len(by_id))
                return None

            row = next(iter(by_id.values()))
            present = self._read_present(path, row)
            state = present.get(VALUE)

            if state == _NULL:
                self._expect_null(path, row, VALUE)
                return None
            if state != _SCALAR:
                self.errors.append('In root row %r: invalid presence state %r for a scalar value.'
                                   % (row[ID], state))
                return None

            return self._cell(path, row, VALUE)

        if node['kind'] == 'object':
            if len(by_id) != 1:
                self.errors.append('Root object table must contain exactly one row, got %d.' % len(by_id))
                return None

            return self._build_object(node, path, next(iter(by_id.values())))

        return self._build_root_array(node, path, by_id)

    def _build_root_array(self, node, path, by_id):
        if 'root' not in by_id:
            self.errors.append('Root array table must contain a container row with %s = "root".' % ID)
            return None

        container = by_id['root']
        present = self._read_present(path, container)
        state = present.get(ROOT_STATE)
        item_rows = [row for row_id, row in by_id.items() if row_id != 'root']

        if state == _NULL:
            if item_rows:
                self.errors.append('Root array is marked null but contains %d element row(s).' % len(item_rows))
            return None
        if state == _EMPTY:
            if item_rows:
                self.errors.append('Root array is marked empty but contains %d element row(s).' % len(item_rows))
            return []
        if state is not None:
            self.errors.append('Root array container row has invalid presence state %r.' % state)
            return None

        if not item_rows:
            self.errors.append('Root array is marked non-empty but has no element rows.')
            return []

        for row in item_rows:
            if row[PARENT] != 'root':
                self.errors.append('Root array element %r has %s=%r, expected "root".'
                                   % (row[ID], PARENT, row[PARENT]))

        return self._build_item_list(node['item'], path, item_rows)

    def _build_object(self, node, path, row):
        present = self._read_present(path, row)
        result = OrderedDict()

        for key, child in node['fields'].items():
            if key not in present:
                # Missing field: omitted from the rebuilt object.
                continue

            state = present[key]
            child_path = path + (key,)
            kind = child['kind']

            if kind == 'scalar':
                if state == _NULL:
                    self._expect_null(path, row, key)
                    result[key] = None
                elif state == _SCALAR:
                    result[key] = self._cell(path, row, key)
                else:
                    self.errors.append('In table %r row %r: invalid state %r for scalar field %r.'
                                       % (self._table_name(path), row[ID], state, key))
                continue

            _, _, grandchildren = self._prepare(child_path)
            child_rows = grandchildren.get(row[ID], [])

            if state == _NULL:
                if child_rows:
                    self.errors.append(
                        'Field %s at %s is marked null but child table %r has %d row(s).'
                        % (key, _render_path(path), self._table_name(child_path), len(child_rows))
                    )
                result[key] = None
                continue

            if kind == 'object':
                if len(child_rows) > 1:
                    self.errors.append('Field %s at %s is an object but %d child rows reference %r.'
                                       % (key, _render_path(path), len(child_rows), row[ID]))
                    result[key] = None
                    continue
                if state == _EMPTY:
                    if child_rows:
                        self.errors.append('Field %s at %s is marked empty but has a child row.'
                                           % (key, _render_path(path)))
                    result[key] = OrderedDict()
                    continue
                if state != _OBJECT:
                    self.errors.append('Field %s at %s has invalid object state %r.'
                                       % (key, _render_path(path), state))
                    result[key] = None
                    continue
                if not child_rows:
                    self.errors.append('Field %s at %s is marked present but its object child row is missing.'
                                       % (key, _render_path(path)))
                    result[key] = None
                    continue

                result[key] = self._build_object(child, child_path, child_rows[0])
            else:
                if state == _EMPTY:
                    if child_rows:
                        self.errors.append('Field %s at %s is marked empty but has %d element row(s).'
                                           % (key, _render_path(path), len(child_rows)))
                    result[key] = []
                    continue
                if state != _ARRAY:
                    self.errors.append('Field %s at %s has invalid array state %r.'
                                       % (key, _render_path(path), state))
                    result[key] = None
                    continue
                if not child_rows:
                    self.errors.append('Field %s at %s is marked non-empty but has no element rows.'
                                       % (key, _render_path(path)))
                    result[key] = None
                    continue

                result[key] = self._build_item_list(child['item'], child_path, child_rows)

        unknown_states = set(present) - set(node['fields'].keys()) - {ROOT_STATE, VALUE}

        if unknown_states:
            self.errors.append('In table %r row %r: %s references unknown field(s) %s.'
                               % (self._table_name(path), row[ID], PRESENT, ', '.join(sorted(unknown_states))))

        self._check_orphan_rows(node, path, row, present)

        return result

    def _check_orphan_rows(self, node, path, row, present):
        """
        Reconcile every container field against its child table, and flag any
        child row that references a nonexistent parent.
        """
        _, parent_by_id, _ = self._prepare(path)

        for key, child in node['fields'].items():
            if child['kind'] == 'scalar':
                continue

            child_path = path + (key,)
            _, child_by_id, grandchildren = self._prepare(child_path)
            child_rows = grandchildren.get(row[ID], [])
            state = present.get(key)

            # A field that is missing or marked null must own no child rows.
            if state in (None, _NULL) and child_rows:
                self.errors.append(
                    'Field %s at %s is missing or null in %s but child table %r contains %d row(s).'
                    % (key, _render_path(path), PRESENT, self._table_name(child_path), len(child_rows))
                )

            # Rows referencing a parent id that does not exist in the parent table.
            for child_row in child_by_id.values():
                if child_row[PARENT] is not None and child_row[PARENT] not in parent_by_id:
                    self.errors.append(
                        'In table %r: row %r references missing parent %r.'
                        % (self._table_name(child_path), child_row[ID], child_row[PARENT])
                    )

    def _expect_null(self, path, row, column):
        if column not in row.keys():
            self.errors.append('In table %r row %r: scalar column %r is missing.'
                               % (self._table_name(path), row[ID], column))
        elif row[column] is not None:
            self.errors.append('Field %r in row %r is marked null but the table holds %r.'
                               % (column, row[ID], row[column]))

    def _cell(self, path, row, column):
        if column not in row.keys():
            self.errors.append('In table %r row %r: scalar column %r is missing.'
                               % (self._table_name(path), row[ID], column))
            return None

        return row[column]

    def _build_item_list(self, item_node, path, item_rows):
        valid_rows = self._check_indices(path, item_rows)

        ordered = sorted(valid_rows, key=lambda row: row[INDEX])
        values = []

        for row in ordered:
            if item_node['kind'] == 'scalar':
                values.append(self._cell(path, row, self._value_column(path)))
            else:
                values.append(self._build_object(item_node, path, row))

        return values

    def _value_column(self, path):
        table = self._table(path)
        candidates = [
            name for name in table.column_names
            if name not in (ID, PARENT, INDEX, PRESENT)
        ]

        if len(candidates) != 1:
            self.errors.append('Scalar array table %r must have exactly one value column, found: %s.'
                               % (self._table_name(path), ', '.join(candidates)))
            return candidates[0] if candidates else VALUE

        return candidates[0]

    def _check_indices(self, path, rows):
        table = self._table(path)

        if INDEX not in table.column_names:
            self.errors.append('Array table %r is missing the %s column.' % (self._table_name(path), INDEX))
            return

        seen = set()
        physical = []
        valid_rows = []

        for row in rows:
            index = row[INDEX]

            if not isinstance(index, Decimal):
                self.errors.append('In table %r row %r: %s must be a number, got %r.'
                                   % (self._table_name(path), row[ID], INDEX, index))
                continue
            if index != index.to_integral_value() or index < 0:
                self.errors.append('In table %r row %r: %s %s is not a non-negative integer.'
                                   % (self._table_name(path), row[ID], INDEX, index))
                continue
            if index in seen:
                self.errors.append('In table %r: duplicate %s %s.' % (self._table_name(path), INDEX, index))
                continue

            seen.add(index)
            physical.append(index)
            valid_rows.append(row)

        if physical != sorted(physical):
            self.errors.append(
                'In table %r: physical row order does not match %s order. Update %s or use NestedTables.reindex().'
                % (self._table_name(path), INDEX, INDEX)
            )

        return valid_rows
