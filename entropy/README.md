## Computation of entropy of silk density

### `utils.py`

### `compute_entropy.py`

Compute entropy of a single web.
Also plot a graph of entropy (with level 0, without level 0, average without level 0) in terms of the number of subdivisions.

Arguments:

- `file_name`
- `num_M`
- `num_levels`
- `min_M`
- `max_M`

### `inference_from_neighbors.py`

Use p-average to infer a silk density from neighbors. Compute (averaged) L^q-error.
Plot a graph of errors in `p` (for a fixed `q`) and find `p` with minimum inference error.

Arguments:

Example:
```
python3 compute_entropy.py --file_name "@011 255 2024-10-04 03-20-37" --num_M 20 --num_levels 10 --min_M 20 --max_M 120
```

- `file_name` numpy file, assumed to be in the `spiderweb/point_clouds` directory.
- `num_M` size of subdivisions
- `p` for `p`-harmonicity
- `q` for L^`q`-error
- `visualize` (boolean flag)
- `plot_p` (boolean flag)
- `delta` for plot
- `max_p` for plot

Example:
```
python3 inference_from_neighbors.py --file_name "@011 255 2024-10-04 03-20-37" --num_M 20 --p 1.0 --q 1.0 --plot_p --delta 0.1 --max_p 1.0
```

