ff := /home/brent/omsf/projects/benchmarking/forcefields/new-tm-2.2.offxml

parameter_id_to_torsion_ids.json: ../scripts/map-parameter-ids-to-torsions.py
	python $<  \
    --input "../datasets/qm/output"             \
    --force-field $(ff)                 \
    --output $@

figs:
	python ../scripts/plot-individual-energy-breakdowns.py  \
    --torsiondrive-id        119466831                                   \
    --output-directory  ../images2                                       \
    --forcefield        $(ff) \
    --qm-dataset        ../datasets/qm/output                           \
    --mm-dataset        ../datasets/mm/minimized-torsiondrive-datasets  \
    --geometry          "mm"
