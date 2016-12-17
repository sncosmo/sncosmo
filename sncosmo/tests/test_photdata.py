import numpy as np
from astropy.table import Table

import sncosmo


def test_select_data():
    data = Table([[1., 2., 3.], ['a', 'b', 'c'], [[1.1, 1.2, 1.3],
                                                  [2.1, 2.2, 2.3],
                                                  [3.1, 3.2, 3.3]]],
                 names=['time', 'x', 'cov'])

    for index in (np.array([True, True, False]),
                  [0, 1],
                  slice(0, 2)):
        selected = sncosmo.select_data(data, index)
        assert np.all(selected['time'] == np.array([1., 2.]))
        assert np.all(selected['x'] == np.array(['a', 'b']))
        assert np.all(selected['cov'] == np.array([[1.1, 1.2], [2.1, 2.2]]))

    # integer indexing switching order
    selected = sncosmo.select_data(data, [2, 0])
    assert np.all(selected['time'] == np.array([3., 1.]))
    assert np.all(selected['x'] == np.array(['c', 'a']))
    assert np.all(selected['cov'] == np.array([[3.3, 3.1], [1.3, 1.1]]))
