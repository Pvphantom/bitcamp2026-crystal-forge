from app.ml.data import LARGE_LATTICE_POINTS, MU_GRID, U_GRID, generate_base_samples, parameter_grid
from app.ml.schema import build_graph_sample


def test_parameter_grid_counts_match_spec() -> None:
    assert len(list(parameter_grid(large_lattice=False))) == len(U_GRID) * len(MU_GRID)
    assert len(list(parameter_grid(large_lattice=True))) == len(LARGE_LATTICE_POINTS)


def test_generate_base_samples_supports_2x2_padding() -> None:
    sample_2x2 = generate_base_samples(
        Lx=2,
        Ly=2,
        max_nodes=6,
        grid=[(0.0, 0.0)],
    )[0]

    assert sample_2x2["nodes"].shape == (6, 5)
    assert sample_2x2["node_mask"].sum().item() == 4


def test_graph_sample_builder_supports_2x3_shapes() -> None:
    sample_2x3 = build_graph_sample(
        Lx=2,
        Ly=3,
        site_features=[
            [1, 0, 0, 0.5, 1],
            [0, 1, 0, -0.5, -1],
            [1, 0, 0, 0.5, 1],
            [0, 1, 0, -0.5, -1],
            [1, 0, 0, 0.5, 1],
            [0, 1, 0, -0.5, -1],
        ],
        bond_strengths={(0, 1): -0.2, (0, 2): -0.2, (1, 3): -0.2, (2, 3): -0.2, (2, 4): -0.2, (3, 5): -0.2, (4, 5): -0.2},
        global_feats=[4.0, 2.0, 6.0],
        label="Antiferromagnet",
        metadata={"sample_id": "2x3"},
        max_nodes=6,
    ).to_dict()
    assert sample_2x3["nodes"].shape == (6, 5)
    assert sample_2x3["node_mask"].sum().item() == 6
