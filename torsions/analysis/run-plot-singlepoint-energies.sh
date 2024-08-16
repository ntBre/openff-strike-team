#!/bin/bash

micromamba activate strike-team-torsions

python ../scripts/plot-qm-vs-mm-profile.py                                  \
    --parameter-id      t4                                                \
    --output-directory  ../images                                           \
    --qm-dataset        ../datasets/qm/output                               \
    --mm-dataset        ../datasets/mm/singlepoint-torsiondrive-datasets    \
    --forcefield        "tm-2.2.offxml"                                     \
    --parameter-ids-to-torsions parameter_id_to_torsion_ids.json            \
    --combination       "~/pydev/qca-datasets-report/combinations/sage-2.2.0.csv" \
    --suffix            "-singlepoint-sage-training"

