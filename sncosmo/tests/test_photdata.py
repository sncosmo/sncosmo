from sncosmo.photdata import alias_map


def test_alias_map():
    mapping = alias_map(['A', 'B_', 'foo'],
                        {'a': set(['a', 'a_']), 'b': set(['b', 'b_'])})
    assert mapping == {'A': 'a', 'B_': 'b'}
