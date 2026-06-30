# Experiment Notes

## Research Questions

1. How does cutoff radius affect runtime, accuracy, and unassigned pixels?
2. Which site distributions create the most mismatches against brute force?
3. How does the frame memory cost grow with site count and grid size?
4. How does the proposed circle-growing approximation compare with JFA?

## Suggested Baselines

Start with CPU runs on small grids:

```powershell
python run_experiment.py --grid 64 --sites 2 --mode random --cutoff 100 --algorithm all --arch cpu --compare-bruteforce --save-images --save-diff
python run_experiment.py --grid 128 --sites 10 --mode boundary --cutoff 100 --algorithm all --arch cpu --compare-bruteforce --save-images --save-diff
```

Then try a moderate GPU run:

```powershell
python run_experiment.py --grid 256 --sites 50 --mode clustered --cutoff 80 --algorithm all --arch gpu --compare-bruteforce --save-images --save-diff
```

## Interpretation Tips

- White pixels in proposed images are unassigned because no site reached them within the cutoff radius.
- Red pixels in difference images are assigned but disagree with brute force.
- If cutoff increases, unassigned pixels should generally decrease while runtime increases.
- Boundary and close-site distributions are useful stress tests for indexing, tie-breaking, and radius quantization.

## Future Improvements

- Add a true MDCS/MSN circle generator as an interchangeable backend.
- Add a midpoint circle plotting mode with absentee-pixel repair.
- Add tiled frame processing to reduce peak memory.
- Add exact-distance tie breaking as an optional analysis mode.
- Add repeated trials per configuration and aggregate confidence intervals.
