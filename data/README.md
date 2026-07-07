# data/

Runtime data for the cache lives here and is gitignored.

- `cold_tier/` : the on-disk cold tier. The server writes one `.npy` file per
  demoted embedding here. It is created automatically on first run.

Nothing in this directory needs to be committed. Delete `cold_tier/` any time to
reset the persistent cache.
