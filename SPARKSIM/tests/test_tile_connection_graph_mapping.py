import numpy as np

from MNSIM.Mapping_Model.Tile_connection_graph import generate_hui_matrix


def test_hui_matrix_preserves_legacy_square_order():
    matrix = generate_hui_matrix(3, 3)

    np.testing.assert_array_equal(
        matrix,
        np.array(
            [
                [0, 1, 8],
                [3, 2, 7],
                [4, 5, 6],
            ]
        ),
    )


def test_hui_matrix_supports_rectangular_tile_arrays():
    matrix = generate_hui_matrix(6, 3)

    assert matrix.shape == (6, 3)
    np.testing.assert_array_equal(np.sort(matrix.ravel()), np.arange(18))

    positions = {
        int(value): tuple(location)
        for value, location in (
            (value, np.argwhere(matrix == value)[0])
            for value in range(18)
        )
    }
    for value in range(17):
        row_delta = abs(positions[value][0] - positions[value + 1][0])
        column_delta = abs(positions[value][1] - positions[value + 1][1])
        assert row_delta + column_delta == 1
