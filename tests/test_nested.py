"""
Tests for the explicit-schema nested-table entry points:
:meth:`.Table.from_nested_object`, :meth:`.Table.from_nested_json` and the
:class:`.NestedTables` container.
"""

import json
import os
import tempfile
from decimal import Decimal
from io import StringIO

from agate import NestedTables, NestedValidationError, Sum, Table
from agate.computations import Formula
from agate.data_types import Boolean, Number, Text
from agate.table import Table as TableClass
from agate.testcase import AgateTestCase

T = Text(cast_nulls=False)


def schema_simple():
    return {
        'id': Number(),
        'name': T,
        'active': Boolean(),
        'note': T,
        'ratio': Number(),
        'address': {'street': T, 'zip': T},
        'empty_obj': {'x': Number()},
        'tags': [T],
        'empty_tags': [T],
        'null_tags': [T],
        'items': [{'sku': T, 'price': Number(), 'qty': Number()}],
    }


def data_simple():
    return {
        'id': 1,
        'name': 'ada',
        'active': True,
        'note': None,
        'ratio': Decimal('1.50'),
        'address': {'street': 'Main St', 'zip': None},
        'empty_obj': {},
        'tags': ['x', 'x', 'y'],
        'empty_tags': [],
        'null_tags': None,
        'items': [
            {'sku': 'a', 'price': Decimal('10.00'), 'qty': 2},
            {'sku': 'b', 'price': Decimal('2.50'), 'qty': 5},
        ],
    }


class TestNestedSplit(AgateTestCase):
    def test_table_methods_are_attached(self):
        self.assertTrue(hasattr(TableClass, 'from_nested_object'))
        self.assertTrue(hasattr(TableClass, 'from_nested_json'))

    def test_split_creates_parent_and_child_tables(self):
        nested = NestedTables.from_object(data_simple(), schema_simple())

        self.assertEqual(
            nested.names(),
            ('root', 'address', 'empty_obj', 'tags', 'empty_tags', 'null_tags', 'items'),
        )
        self.assertIs(nested.root, nested['root'])

        root = nested.root
        self.assertColumnNames(root, ('__id', '__parent', '__present', 'id', 'name', 'active', 'note', 'ratio'))
        self.assertEqual(len(root.rows), 1)
        self.assertEqual(root.rows[0]['__id'], 'root')
        self.assertIsNone(root.rows[0]['__parent'])

        # Object child table.
        address = nested['address']
        self.assertColumnNames(address, ('__id', '__parent', '__present', 'street', 'zip'))
        self.assertEqual(address.rows[0]['__parent'], 'root')

        # Array tables carry __index; object arrays carry the item's scalar columns.
        items = nested['items']
        self.assertColumnNames(items, ('__id', '__parent', '__index', '__present', 'sku', 'price', 'qty'))
        self.assertEqual([r['__index'] for r in items.rows], [Decimal('0'), Decimal('1')])

        # Scalar arrays have a single value column named after the field.
        tags = nested['tags']
        self.assertEqual(tags.column_names[-1], 'tags')
        self.assertEqual([r['tags'] for r in tags.rows], ['x', 'x', 'y'])

    def test_empty_containers_still_get_tables(self):
        nested = NestedTables.from_object(data_simple(), schema_simple())

        self.assertEqual(len(nested['empty_tags'].rows), 0)
        self.assertEqual(len(nested['null_tags'].rows), 0)
        self.assertEqual(len(nested['empty_obj'].rows), 0)
        self.assertIn('__index', nested['empty_tags'].column_names)

    def test_missing_null_empty_are_distinct(self):
        nested = NestedTables.from_object(data_simple(), schema_simple())
        present = json.loads(nested.root.rows[0]['__present'])

        self.assertEqual(present['note'], 'null')
        self.assertEqual(present['empty_obj'], 'empty')
        self.assertEqual(present['empty_tags'], 'empty')
        self.assertEqual(present['null_tags'], 'null')
        self.assertEqual(present['ratio'], 'scalar')
        # A field absent from the data is absent from the presence map entirely.
        self.assertNotIn('absent', present)

    def test_duplicate_children_and_slash_keys_preserved(self):
        data = {
            'a/b': 1,
            'a': {'b': 2},
            'dup': [{'k': 'v'}, {'k': 'v'}],
        }
        schema = {
            'a/b': Number(),
            'a': {'b': Number()},
            'dup': [{'k': T}],
        }
        nested = NestedTables.from_object(data, schema)

        self.assertEqual(len(nested['dup'].rows), 2)
        self.assertEqual(nested.to_object(), data)

    def test_parallel_arrays_are_not_multiplied(self):
        data = {'a': [1, 2, 3], 'b': [10, 20]}
        nested = NestedTables.from_object(data, {'a': [Number()], 'b': [Number()]})

        self.assertEqual(len(nested['a'].rows), 3)
        self.assertEqual(len(nested['b'].rows), 2)
        self.assertEqual(nested.to_object(), data)

    def test_object_columns_can_compute(self):
        nested = NestedTables.from_object(data_simple(), schema_simple())
        computed = nested.root.compute([
            ('name_upper', Formula(T, lambda r: r['name'].upper())),
        ])

        self.assertEqual(computed.rows[0]['name_upper'], 'ADA')

    def test_child_tables_are_independently_analyzable(self):
        nested = NestedTables.from_object(data_simple(), schema_simple())

        self.assertEqual(nested['items'].aggregate(Sum('price')), Decimal('12.50'))
        filtered = nested['items'].where(lambda r: r['sku'] == 'a')
        self.assertEqual(len(filtered.rows), 1)


class TestNestedTypes(AgateTestCase):
    def test_strict_inference_keeps_strings_and_booleans(self):
        rows = [
            {'s_num': '12345', 's_empty': '', 's_word': 'null', 'flag': True, 'n': Decimal('0.1')},
            {'s_num': '678', 's_empty': 'a', 's_word': 'b', 'flag': False, 'n': Decimal('2')},
        ]
        nested = NestedTables.from_object(rows)
        types = dict(zip(nested.root.column_names, nested.root.column_types))

        self.assertIsInstance(types['s_num'], Text)
        self.assertIsInstance(types['s_empty'], Text)
        self.assertIsInstance(types['s_word'], Text)
        self.assertIsInstance(types['flag'], Boolean)
        self.assertIsInstance(types['n'], Number)

        elements = [r for r in nested.root.rows if r['__id'] != 'root']
        self.assertEqual(elements[0]['s_empty'], '')
        self.assertEqual(elements[0]['s_word'], 'null')
        self.assertEqual(nested.to_object(), rows)

    def test_inferred_bool_number_conflict_fails(self):
        with self.assertRaises(NestedValidationError) as ctx:
            NestedTables.from_object([{'x': True}, {'x': 1}])

        self.assertIn('conflicting scalar types', ctx.exception.messages[0])

    def test_explicit_boolean_is_not_coerced_to_number(self):
        with self.assertRaises(NestedValidationError):
            NestedTables.from_object({'b': True}, {'b': Number()})

    def test_decimal_roundtrip_is_exact(self):
        data = {
            'big': Decimal('12345678901234567890.123456789'),
            'small': Decimal('0.000000001'),
            'sci': Decimal('1E+3'),
        }
        nested = NestedTables.from_object(data, {'big': Number(), 'small': Number(), 'sci': Number()})
        rebuilt = nested.to_object()

        self.assertEqual(rebuilt, data)

        buf = StringIO()
        nested.to_json(buf)
        reloaded = json.loads(buf.getvalue(), parse_float=Decimal)
        self.assertEqual(reloaded['big'], Decimal('12345678901234567890.123456789'))
        self.assertEqual(reloaded['small'], Decimal('0.000000001'))
        self.assertEqual(reloaded['sci'], Decimal('1000'))

    def test_array_item_missing_vs_null_distinct(self):
        data = {'items': [
            {'sku': 'a'},
            {'sku': 'b', 'note': None},
        ]}
        schema = {'items': [{'sku': T, 'note': T}]}
        nested = NestedTables.from_object(data, schema)

        p0 = json.loads(nested['items'].rows[0]['__present'])
        p1 = json.loads(nested['items'].rows[1]['__present'])
        self.assertNotIn('note', p0)
        self.assertEqual(p1['note'], 'null')
        self.assertEqual(nested.to_object(), data)


class TestNestedRebuild(AgateTestCase):
    def _nested(self):
        return NestedTables.from_object(data_simple(), schema_simple())

    def test_roundtrip(self):
        nested = self._nested()
        self.assertEqual(nested.to_object(), data_simple())

    def test_filter_array_children(self):
        nested = self._nested()
        kept = nested.reindex(nested['items'].where(lambda r: r['sku'] != 'b'))
        replaced = nested.replace({
            name: (kept if name == 'items' else nested[name]) for name in nested.names()
        })

        self.assertEqual([i['sku'] for i in replaced.to_object()['items']], ['a'])
        # Original hierarchy is untouched.
        self.assertEqual([i['sku'] for i in nested.to_object()['items']], ['a', 'b', ])

    def test_reorder_array_children(self):
        nested = self._nested()
        reordered = nested.reindex(nested['items'].order_by('qty'))
        replaced = nested.replace({
            name: (reordered if name == 'items' else nested[name]) for name in nested.names()
        })

        self.assertEqual([i['sku'] for i in replaced.to_object()['items']], ['a', 'b'])
        self.assertEqual([i['qty'] for i in replaced.to_object()['items']], [2, 5])

    def test_add_element_to_empty_array(self):
        nested = NestedTables.from_object(
            {'tags': [], 'name': 'n'},
            {'tags': [T], 'name': T},
        )
        added = Table(
            [('new1', 'root', Decimal(0), '{}', 'fresh')],
            list(nested['tags'].column_names),
            list(nested['tags'].column_types),
        )
        root = nested.root
        present = json.loads(root.rows[0]['__present'])
        present['tags'] = 'array'
        new_root = self._with_present(root, present)

        replaced = nested.replace({'root': new_root, 'tags': added})
        self.assertEqual(replaced.to_object()['tags'], ['fresh'])

    def test_populate_empty_object(self):
        nested = NestedTables.from_object({'o': {}}, {'o': {'k': Number()}})
        child = Table(
            [('n100', 'root', json.dumps({'k': 'scalar'}), Decimal('7'))],
            list(nested['o'].column_names),
            list(nested['o'].column_types),
        )
        root = nested.root
        present = json.loads(root.rows[0]['__present'])
        present['o'] = 'object'
        new_root = self._with_present(root, present)

        replaced = nested.replace({'root': new_root, 'o': child})
        self.assertEqual(replaced.to_object(), {'o': {'k': Decimal('7')}})

    def test_null_to_empty_array_transition(self):
        nested = NestedTables.from_object({'v': None}, {'v': [Number()]})
        root = nested.root
        present = json.loads(root.rows[0]['__present'])
        present['v'] = 'empty'
        new_root = self._with_present(root, present)

        replaced = nested.replace({'root': new_root, 'v': nested['v']})
        self.assertEqual(replaced.to_object(), {'v': []})

    def test_computed_extra_column_is_ignored_on_rebuild(self):
        nested = NestedTables.from_object(
            {'name': 'abc', 'tags': [1]},
            {'name': T, 'tags': [Number()]},
        )
        computed = nested.root.compute([('name_len', Formula(Number(), lambda r: len(r['name'])))])
        replaced = nested.replace({
            name: (computed if name == 'root' else nested[name]) for name in nested.names()
        })
        self.assertEqual(replaced.to_object(), {'name': 'abc', 'tags': [Decimal('1')]})

    def test_rebuild_reads_current_table_not_template(self):
        nested = self._nested()
        # Mutate a scalar value in the table; the rebuild must reflect it.
        root = nested.root
        rows = []
        for r in root.rows:
            values = list(r.values())
            values[root.column_names.index('name')] = 'mutated'
            rows.append(tuple(values))
        changed = Table(rows, list(root.column_names), list(root.column_types))
        replaced = nested.replace({
            name: (changed if name == 'root' else nested[name]) for name in nested.names()
        })
        self.assertEqual(replaced.to_object()['name'], 'mutated')

    def _with_present(self, table, present):
        rows = []
        raw = json.dumps(present)
        pos = table.column_names.index('__present')

        for r in table.rows:
            values = list(r.values())
            values[pos] = raw
            rows.append(tuple(values))

        return Table(rows, list(table.column_names), list(table.column_types))

    def _with_column(self, table, column, value, row_index=0):
        pos = table.column_names.index(column)
        rows = []

        for i, r in enumerate(table.rows):
            values = list(r.values())
            if i == row_index:
                values[pos] = value
            rows.append(tuple(values))

        return Table(rows, list(table.column_names), list(table.column_types))

    def test_dangling_parent_fails(self):
        nested = self._nested()
        changed = self._with_column(nested['items'], '__parent', 'ghost', row_index=0)
        tables = {name: (changed if name == 'items' else nested[name]) for name in nested.names()}

        with self.assertRaises(NestedValidationError) as ctx:
            nested.replace(tables).to_object()
        self.assertTrue(any('missing parent' in m for m in ctx.exception.messages))
        self.assertEqual(nested.to_object(), data_simple())

    def test_duplicate_index_fails(self):
        self._assert_index_error(1, Decimal(0), 'duplicate __index')

    def test_negative_index_fails(self):
        self._assert_index_error(0, Decimal('-1'), 'non-negative integer')

    def test_order_mismatch_fails(self):
        nested = self._nested()
        # Reverse the physical rows without renumbering __index, so the index
        # column no longer matches physical order.
        tables = {
            name: (nested[name].order_by('qty', reverse=True) if name == 'items' else nested[name])
            for name in nested.names()
        }
        with self.assertRaises(NestedValidationError) as ctx:
            nested.replace(tables).to_object()
        self.assertTrue(any('physical row order' in m for m in ctx.exception.messages))

    def test_null_state_with_rows_fails(self):
        nested = self._nested()
        root = nested.root
        present = json.loads(root.rows[0]['__present'])
        present['tags'] = 'null'
        new_root = self._with_present(root, present)
        tables = {name: (new_root if name == 'root' else nested[name]) for name in nested.names()}

        with self.assertRaises(NestedValidationError) as ctx:
            nested.replace(tables).to_object()
        self.assertTrue(any('marked null' in m for m in ctx.exception.messages))

    def test_empty_state_with_rows_fails(self):
        nested = self._nested()
        root = nested.root
        present = json.loads(root.rows[0]['__present'])
        present['tags'] = 'empty'
        new_root = self._with_present(root, present)
        tables = {name: (new_root if name == 'root' else nested[name]) for name in nested.names()}

        with self.assertRaises(NestedValidationError) as ctx:
            nested.replace(tables).to_object()
        self.assertTrue(any('marked empty' in m for m in ctx.exception.messages))

    def test_nonempty_state_without_rows_fails(self):
        nested = self._nested()
        empty = Table(
            [], list(nested['tags'].column_names), list(nested['tags'].column_types)
        )
        tables = {name: (empty if name == 'tags' else nested[name]) for name in nested.names()}

        with self.assertRaises(NestedValidationError) as ctx:
            nested.replace(tables).to_object()
        self.assertTrue(any('non-empty but has no element' in m for m in ctx.exception.messages))

    def test_bad_present_json_fails(self):
        nested = self._nested()
        root = nested.root
        pos = root.column_names.index('__present')
        rows = []
        for r in root.rows:
            values = list(r.values())
            values[pos] = '{broken'
            rows.append(tuple(values))
        bad_root = Table(rows, list(root.column_names), list(root.column_types))
        tables = {name: (bad_root if name == 'root' else nested[name]) for name in nested.names()}

        with self.assertRaises(NestedValidationError) as ctx:
            nested.replace(tables).to_object()
        self.assertTrue(any('not valid JSON' in m for m in ctx.exception.messages))

    def test_duplicate_id_fails(self):
        nested = self._nested()
        table = nested['items']
        rows = [tuple(r) for r in table.rows]
        duplicate = list(rows[1])
        duplicate[0] = rows[0][0]
        rows[1] = tuple(duplicate)
        changed = Table(rows, list(table.column_names), list(table.column_types))
        tables = {name: (changed if name == 'items' else nested[name]) for name in nested.names()}

        with self.assertRaises(NestedValidationError) as ctx:
            nested.replace(tables).to_object()
        self.assertTrue(any('duplicate __id' in m for m in ctx.exception.messages))

    def test_unknown_state_token_fails(self):
        nested = self._nested()
        root = nested.root
        present = json.loads(root.rows[0]['__present'])
        present['tags'] = 'bogus'
        new_root = self._with_present(root, present)
        tables = {name: (new_root if name == 'root' else nested[name]) for name in nested.names()}

        with self.assertRaises(NestedValidationError):
            nested.replace(tables).to_object()

    def _assert_index_error(self, row_index, index_value, substring):
        nested = self._nested()
        changed = self._with_column(nested['items'], '__index', index_value, row_index=row_index)
        tables = {name: (changed if name == 'items' else nested[name]) for name in nested.names()}

        with self.assertRaises(NestedValidationError) as ctx:
            nested.replace(tables).to_object()
        self.assertTrue(any(substring in m for m in ctx.exception.messages))


class TestNestedValidation(AgateTestCase):
    def test_reserved_keys_rejected(self):
        for key in ('__id', '__parent', '__index', '__present', '__value'):
            with self.assertRaises(NestedValidationError) as ctx:
                NestedTables.from_object({key: 1}, {key: Number()})
            self.assertIn('reserved', ctx.exception.messages[0])

    def test_array_schema_must_have_one_item(self):
        with self.assertRaises(NestedValidationError) as ctx:
            NestedTables.from_object([1, 2], [Number(), T])
        self.assertIn('exactly one item schema', ctx.exception.messages[0])

    def test_bad_schema_node_rejected(self):
        with self.assertRaises(NestedValidationError) as ctx:
            NestedTables.from_object({'a': 1}, {'a': 42})
        self.assertIn('must be a DataType', ctx.exception.messages[0])

    def test_extra_field_rejected(self):
        with self.assertRaises(NestedValidationError) as ctx:
            NestedTables.from_object({'a': 1, 'b': 2}, {'a': Number()})
        self.assertIn('fields not in the schema', ctx.exception.messages[0])

    def test_cast_error_reported(self):
        with self.assertRaises(NestedValidationError) as ctx:
            NestedTables.from_object({'n': 'abc'}, {'n': Number()})
        self.assertIn('Can not parse value', ctx.exception.messages[0])

    def test_replace_rejects_missing_and_unknown_tables(self):
        nested = NestedTables.from_object({'a': [1]}, {'a': [Number()]})

        with self.assertRaises(NestedValidationError) as ctx:
            nested.replace({'root': nested.root})
        self.assertIn('Missing required table', ctx.exception.messages[0])

        with self.assertRaises(NestedValidationError) as ctx:
            extra = dict(nested.tables)
            extra['ghost'] = nested.root
            nested.replace(extra)
        self.assertIn('Unknown table', ctx.exception.messages[0])

    def test_validate_passes_on_clean_hierarchy(self):
        self._nested().validate()

    def _nested(self):
        return NestedTables.from_object(data_simple(), schema_simple())


class TestNestedRootShapes(AgateTestCase):
    def test_root_scalar(self):
        for value in (Decimal('3.14'), 'hello', True, None):
            self.assertEqual(NestedTables.from_object(value).to_object(), value)

    def test_root_array_shapes(self):
        for value in ([1, 2, 3], [], None):
            nested = NestedTables.from_object(value)
            self.assertEqual(nested.to_object(), value)

    def test_root_array_scalar_null_elements(self):
        nested = NestedTables.from_object([1, None, 3], [Number()])
        self.assertEqual(nested.to_object(), [Decimal('1'), None, Decimal('3')])


class TestNestedJSON(AgateTestCase):
    def setUp(self):
        self.text = json.dumps({'items': [{'p': 1.5}], 'name': 'n'})
        self.schema = {'items': [{'p': Number()}], 'name': T}
        self.expected = {'items': [{'p': Decimal('1.5')}], 'name': 'n'}

    def test_from_nested_json_file_like(self):
        nested = Table.from_nested_json(StringIO(self.text), schema=self.schema)
        self.assertEqual(nested.to_object(), self.expected)

    def test_from_nested_json_path(self):
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        try:
            with open(path, 'w') as f:
                f.write(self.text)

            nested = Table.from_nested_json(path, schema=self.schema)
            self.assertEqual(nested.to_object(), self.expected)
        finally:
            os.remove(path)

    def test_from_nested_object_entry(self):
        nested = Table.from_nested_object(self.expected, schema=self.schema)
        self.assertIsInstance(nested, NestedTables)

    def test_to_json_indent_and_unicode(self):
        data = {'emoji': '👍', 'n': Decimal('3.0')}
        nested = NestedTables.from_object(data, {'emoji': T, 'n': Number()})

        buf = StringIO()
        nested.to_json(buf, indent=2)
        self.assertIn('👍', buf.getvalue())
        self.assertEqual(json.loads(buf.getvalue(), parse_float=Decimal), data)


class TestNestedDeepHierarchy(AgateTestCase):
    def setUp(self):
        self.data = {
            'orders': [
                {'id': 1, 'customer': {'name': 'a', 'addr': {'city': 'NYC'}},
                 'lines': [{'sku': 'x', 'qty': Decimal('2')},
                           {'sku': 'y', 'qty': Decimal('1')}]},
                {'id': 2, 'customer': {'name': 'b', 'addr': {'city': 'LA'}},
                 'lines': [{'sku': 'z', 'qty': Decimal('9')}]},
            ],
        }
        self.schema = {
            'orders': [{
                'id': Number(),
                'customer': {'name': T, 'addr': {'city': T}},
                'lines': [{'sku': T, 'qty': Number()}],
            }],
        }

    def test_deep_roundtrip_and_table_names(self):
        nested = NestedTables.from_object(self.data, self.schema)

        self.assertEqual(
            nested.names(),
            ('root', 'orders', 'orders/customer', 'orders/customer/addr', 'orders/lines'),
        )
        self.assertEqual(nested.to_object(), self.data)

    def test_deep_parent_linkage(self):
        nested = NestedTables.from_object(self.data, self.schema)
        order_ids = {r['__id'] for r in nested['orders'].rows if r['__id'] != 'root'}

        self.assertTrue({r['__parent'] for r in nested['orders/lines'].rows} <= order_ids)

    def test_deep_child_filter(self):
        nested = NestedTables.from_object(self.data, self.schema)
        kept = nested.reindex(nested['orders/lines'].where(lambda r: r['sku'] != 'y'))
        replaced = nested.replace({
            name: (kept if name == 'orders/lines' else nested[name]) for name in nested.names()
        })
        result = replaced.to_object()

        self.assertEqual([l['sku'] for l in result['orders'][0]['lines']], ['x'])
        self.assertEqual([l['sku'] for l in result['orders'][1]['lines']], ['z'])
        # The original hierarchy is unchanged.
        self.assertEqual(nested.to_object(), self.data)

    def test_shared_child_table_partitions_by_parent(self):
        data = {'items': [{'meta': {'k': 1}}, {'meta': {'k': 2}}]}
        schema = {'items': [{'meta': {'k': Number()}}]}
        nested = NestedTables.from_object(data, schema)

        self.assertEqual(len(nested['items/meta'].rows), 2)
        self.assertEqual(nested.to_object(), data)


class TestNestedMalformedTables(AgateTestCase):
    def _nested(self):
        return NestedTables.from_object(
            {'items': [{'p': Decimal('1')}], 'tags': ['a'], 'name': 'x'},
            {'items': [{'p': Number()}], 'tags': [T], 'name': T},
        )

    def test_dropped_scalar_column_reports_error(self):
        nested = self._nested()
        broken = nested['items'].select(('__id', '__parent', '__index', '__present'))
        with self.assertRaises(NestedValidationError) as ctx:
            nested.replace({n: (broken if n == 'items' else nested[n])
                            for n in nested.names()}).to_object()
        self.assertTrue(any("scalar column 'p' is missing" in m for m in ctx.exception.messages))

    def test_dropped_value_column_in_scalar_array_reports_error(self):
        nested = self._nested()
        broken = nested['tags'].select(('__id', '__parent', '__index', '__present'))
        with self.assertRaises(NestedValidationError) as ctx:
            nested.replace({n: (broken if n == 'tags' else nested[n])
                            for n in nested.names()}).to_object()
        self.assertTrue(any('exactly one value column' in m for m in ctx.exception.messages))

    def test_dropped_system_column_reports_error(self):
        nested = self._nested()
        broken = nested.root.select(tuple(c for c in nested.root.column_names if c != '__present'))
        with self.assertRaises(NestedValidationError) as ctx:
            nested.replace({n: (broken if n == 'root' else nested[n])
                            for n in nested.names()}).to_object()
        self.assertTrue(any('missing required system column' in m for m in ctx.exception.messages))

    def test_wrong_index_type_reports_error(self):
        nested = self._nested()
        rows = [tuple(r) for r in nested['items'].rows]
        broken = Table(
            rows,
            list(nested['items'].column_names),
            [T, T, T, T, Number()],
        )
        with self.assertRaises(NestedValidationError) as ctx:
            nested.replace({n: (broken if n == 'items' else nested[n])
                            for n in nested.names()}).to_object()
        self.assertTrue(any('must have type Number' in m for m in ctx.exception.messages))

    def test_errors_are_aggregated(self):
        nested = NestedTables.from_object(
            {'a': [1, 2], 'b': [10]},
            {'a': [Number()], 'b': [Number()]},
        )

        def break_table(table, parent_token):
            rows = [list(r) for r in table.rows]
            rows[0][1] = parent_token
            return Table([tuple(r) for r in rows], list(table.column_names), list(table.column_types))

        broken = {
            'root': nested.root,
            'a': break_table(nested['a'], 'ghost-a'),
            'b': break_table(nested['b'], 'ghost-b'),
        }
        with self.assertRaises(NestedValidationError) as ctx:
            nested.replace(broken).to_object()

        self.assertGreaterEqual(len(ctx.exception.messages), 2)


class TestJsonDumps(AgateTestCase):
    def test_matches_stdlib_for_plain_values(self):
        from agate.nested import json_dumps

        cases = [
            {'a': None, 'b': [], 'c': {}, 'd': [1, {'x': 'y'}, None, True, False]},
            {'unicode键': '值\n\t"quoted"', 'emoji': '😀'},
            {}, [], '', 'plain', 42, 3.5, True, False, None,
        ]
        for value in cases:
            self.assertEqual(json_dumps(value), json.dumps(value, ensure_ascii=False))

    def test_decimal_is_exact_number(self):
        from agate.nested import json_dumps

        for value, expected in [
            (Decimal('1.50'), '1.50'),
            (Decimal('0'), '0'),
            (Decimal('-3.14'), '-3.14'),
            (Decimal('1E+3'), '1000'),
            (Decimal('100.00'), '100.00'),
        ]:
            self.assertEqual(json_dumps(value), expected)

    def test_mixed_decimal_roundtrip(self):
        from agate.nested import json_dumps

        value = {'n': [Decimal('1.50')], 's': 'x'}
        self.assertEqual(
            json.loads(json_dumps(value), parse_float=Decimal),
            {'n': [Decimal('1.50')], 's': 'x'},
        )

    def test_unsupported_type_raises(self):
        from agate.nested import json_dumps

        with self.assertRaises(TypeError):
            json_dumps({'x': object()})
