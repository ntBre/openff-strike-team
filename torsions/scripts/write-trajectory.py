import typing

import click
from openff.toolkit import Molecule
import MDAnalysis as mda
from MDAnalysis.analysis import align
import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds


@click.command()
@click.option(
    "--torsiondrive-id",
    type=int,
    help="The torsiondrive id to use.",
    required=True
)
@click.option(
    "--dataset",
    "input_dataset_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
    help="The path to the dataset with coordinates.",
    default="datasets/mm/minimized-torsiondrive-datasets",
    required=True
)
@click.option(
    "--coordinate-type",
    type=click.Choice(["qm", "mm"]),
    help="The coordinate type to use.",
    required=True
)
@click.option(
    "--output-path",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    help="The path to save the trajectory.",
    default="trajectory.pdb",
    required=True
)
def main(
    torsiondrive_id: int,
    input_dataset_path: str,
    coordinate_type: typing.Literal["qm", "mm"],
    output_path: str,
):
    dataset = ds.dataset(input_dataset_path)
    expression = pc.field("torsiondrive_id") == torsiondrive_id
    dataset = dataset.filter(expression)

    coordinate_column = f"{coordinate_type}_coordinates"

    columns = [
        "grid_id",
        "mapped_smiles",
        coordinate_column,
    ]
    rows = sorted(
        dataset.to_table(columns=columns).to_pylist(),
        key=lambda x: x["grid_id"],
    )

    mol = Molecule.from_mapped_smiles(
        rows[0]["mapped_smiles"],
        allow_undefined_stereo=True,
    )
    u = mda.Universe(mol.to_rdkit())
    coordinates = []
    for row in rows:
        coordinates.append(
            np.array(row[coordinate_column]).reshape((-1, 3))
        )
    u.load_new(np.array(coordinates))

    # align to first three atoms of dihedral
    dihedral = dataset.to_table(columns=["dihedral"]).to_pandas().iloc[0]["dihedral"]
    sel_str = f"index {' '.join(map(str, dihedral[:3]))}"
    align.AlignTraj(u, u, select=sel_str, in_memory=True).run()

    with mda.Writer(output_path, n_atoms=u.atoms.n_atoms) as w:
        for ts in u.trajectory:
            w.write(u.atoms)
    print(f"Saved trajectory to {output_path}")


if __name__ == "__main__":
    main()
