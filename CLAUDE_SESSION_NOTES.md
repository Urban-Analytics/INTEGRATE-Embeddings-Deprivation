# Claude Session Notes — Script Review & Refactoring

## Current Task
Going through each notebook script in this directory and:
1. Refactoring to use the H5 data store where appropriate
2. Fixing bugs
3. Improving readability, reliability, and correctness

Scripts 2-7 are done. **Script 8 is next.**

## H5 Data Store
- **H5 path**: `../../../../data/embeddings/sample_points_cache/street_data.h5` (relative to this dir)
- **Config**: `directory_filepaths.py` holds shared paths (`data_dir`, `h5_filename`, `lsoas_file`, `imd_file`)

### H5 Datasets
- `point_id` (18,897 points), `latitude`, `longitude`
- `images_jpeg` (N x 4) - variable-length JPEG bytes
- `images_present` (N x 4) - boolean array
- `embeddings_clip` (N x 4 x 512) - CLIP embeddings, NaN if missing
- `image_paths` (|S512 fixed-length byte strings), `date`

### CRITICAL: point_id != H5 row index
- H5 has 18,897 rows (indices 0..18896)
- `point_id` values range 0..19195 with 299 gaps (from upstream filtering)
- **Never use point_id directly as an H5 array index**
- To map: `pid_to_h5row = {int(pid): i for i, pid in enumerate(f["point_id"][:])}`
- `image_paths` need `.decode("ascii").rstrip("\x00")` to compare (null-padded)

## Script Status

| Script | Status | Notes |
|--------|--------|-------|
| 1-SampleStreetNetwork | NOT IN SCOPE | Creates the H5 store from source pickle |
| 2-CalculateEmbeddings | DONE | Loads images from H5, writes CLIP embeddings back |
| 3-ProcessEmbeddings+FindMedianEmbeddingPerLSOA | DONE | Reads from H5, spatial join, expands one-row-per-image, saves pickle + per-LSOA summaries |
| 4-RunModelsWithMedianEmbedding | DONE | XGBoost with IMD data, bug fixes applied |
| 5-IdentifyOptimalClusterNumber | DONE | Loads pickle from script 3, clustering, image display from H5 (with pid lookup), saves cluster assignments |
| 6-TestModelOverClusters_ControlledForSampleSize | DONE | Fixed: pickle path, IMD path, model cloning, memory issue (slim_gdf + n_jobs=8), results caching |
| 7-FindMedianEmbeddings_ForEachOf7Clusters | DONE | Fixed: pickle path, save path, cleaned imports, updated markdown |
| 8-RunModels_ForEachOf7Clusters | TODO | Next to review |

## Recurring Bugs Found and Fixed
These same bugs appeared in multiple scripts — check for them in script 8:
1. **Double embeddings path**: `data_dir + "embeddings/..."` produces `data/embeddings/embeddings/...` because `data_dir` already includes `embeddings/`. Fix: `os.path.join(data_dir, "filename.pkl")`.
2. **Hardcoded wrong IMD path**: Scripts had `os.path.join("../../../../", "data", "imd", ...)` but the 2025 IMD file is at `data/embeddings/imd/`. Fix: use `imd_file` from `directory_filepaths.py`.
3. **`pickle.load` instead of `pd.read_pickle`**: Readability fix.

## Key Data Flow
- Script 3 produces `one_row_per_image_cleaned.pkl` (75,586 rows)
- Script 5 adds cluster columns → `one_row_per_image_cleaned_with_cluster_numbers.pkl`
- Scripts 6, 7, 8 consume the script 5 pickle
- Script 7 produces `per_lsoa_embedding_summaries/median_embedding_per_cluster.pkl`
- User is OK keeping intermediate pickle files

## User Preferences
- Don't delete commented-out code cells unless asked
- Include comments in changes
- Add markdown cells where helpful
- Summarise plan before making changes
- Check markdown descriptions match actual code
