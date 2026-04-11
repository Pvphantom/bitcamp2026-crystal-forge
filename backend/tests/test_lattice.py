from app.physics.lattice import nn_bonds, q_dn, q_up, site_index, site_major_to_spin_major_layout


def test_site_index_row_major() -> None:
    assert site_index(0, 0, 2) == 0
    assert site_index(1, 0, 2) == 1
    assert site_index(0, 1, 2) == 2
    assert site_index(1, 1, 2) == 3


def test_nn_bonds_2x2() -> None:
    assert nn_bonds(2, 2) == [(0, 1), (0, 2), (1, 3), (2, 3)]


def test_nn_bonds_2x3() -> None:
    assert nn_bonds(2, 3) == [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)]


def test_spin_major_qubit_helpers() -> None:
    Ns = 4
    assert [q_up(i) for i in range(Ns)] == [0, 1, 2, 3]
    assert [q_dn(i, Ns) for i in range(Ns)] == [4, 5, 6, 7]


def test_site_major_to_spin_major_layout() -> None:
    assert site_major_to_spin_major_layout(4) == [0, 4, 1, 5, 2, 6, 3, 7]
