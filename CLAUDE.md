# Claude Session Notes — Script Review & Refactoring

## Current Task
All notebook scripts (2–9b) have been reviewed and refactored. Now updating the LaTeX paper (`latex/main.tex`) to align with the notebook results.

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
| 7-RunModels_ForEachOfNClusters | DONE | Merged old scripts 7+8: computes per-LSOA embedding summaries then trains per-cluster XGBoost models. Removed intermediate pickle (data flows in memory). |
| 8-Prediction_Deprivation_domains | DONE | Predicts each IMD domain (not just overall) per cluster + global. Default metric is NRMSE (comparable across clusters). Uses rank; TODO: switch to score and make consistent with scripts 4 & 7. |
| 9a-DownloadAlphaEarthEmbeddings | DONE | Rewritten: exports GeoTIFF to Drive, then computes zonal stats locally with rasterstats. No more per-batch EE API calls. |
| 9b-RunModelWithAlphaEarthEmbeddings | DONE | XGBoost on AlphaEarth 64-dim satellite embeddings. CV R²=0.306 vs street-view 0.660. Saves model + residuals map. |

## Recurring Bugs Found and Fixed
These same bugs appeared in multiple scripts — check for them in scripts 9a/9b:
1. **Double embeddings path**: `data_dir + "embeddings/..."` produces `data/embeddings/embeddings/...` because `data_dir` already includes `embeddings/`. Fix: `os.path.join(data_dir, "filename.pkl")`.
2. **Hardcoded wrong IMD path**: Scripts had `os.path.join("../../../../", "data", "imd", ...)` but the 2025 IMD file is at `data/embeddings/imd/`. Fix: use `imd_file` from `directory_filepaths.py`.
3. **`pickle.load` instead of `pd.read_pickle`**: Readability fix.

## Key Data Flow
- Script 3 produces `one_row_per_image_cleaned.pkl` (75,586 rows)
- Script 5 adds cluster columns → `one_row_per_image_cleaned_with_cluster_numbers.pkl`
- Scripts 6, 7, and 8 consume the script 5 pickle
- Script 7 (merged) computes summaries in memory and trains per-cluster models (no intermediate pickle)
- User is OK keeping intermediate pickle files

## User Preferences
- Don't delete commented-out code cells unless asked
- Include comments in changes
- Add markdown cells where helpful
- Summarise plan before making changes
- Check markdown descriptions match actual code
