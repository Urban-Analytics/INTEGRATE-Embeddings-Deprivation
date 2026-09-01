"""
Spatial autocorrelation diagnostics and spatial cross-validation helpers.

Used by notebook 11-SpatialAutocorrelationChecks. Kept in a module (rather than
in the notebook) so that scripts 6, 7, 8 and 9b can reuse the same weights
matrix, the same Moran's I calls and the same splitter if the diagnostics turn
out to matter.
"""

import numpy as np
import pandas as pd

from esda.moran import Moran
from libpysal.weights import KNN, Queen, attach_islands
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr


# --------------------------------------------------------------------------
# Grouping: local authority districts as spatial blocks
# --------------------------------------------------------------------------

def lad_groups(names, name_col="LSOA21NM"):
    """Derive the local authority district (LAD) name from LSOA names.

    LSOA names take the form "Manchester 001A" — the LAD name followed by a
    number and a letter — so the LAD is everything up to the final space.

    Greater Manchester is made up of exactly 10 LADs, which makes
    leave-one-LAD-out a natural (and completely spatial) blocking scheme that
    needs no extra data and no arbitrary choice of block size.

    Parameters
    ----------
    names : pandas.Series or geopandas.GeoDataFrame
        Either the LSOA name column itself, or a frame containing `name_col`.

    Returns
    -------
    pandas.Series of LAD names, aligned to the input.
    """
    if hasattr(names, "columns"):          # a DataFrame was passed
        names = names[name_col]
    return names.str.rsplit(" ", n=1).str[0]


# --------------------------------------------------------------------------
# Spatial weights
# --------------------------------------------------------------------------

def build_weights(gdf, transform="r", verbose=True):
    """Build a row-standardised queen-contiguity spatial weights matrix.

    Queen contiguity (sharing an edge *or* a corner) is the standard choice for
    a complete, non-overlapping set of polygons like LSOAs.

    Any polygon with no contiguous neighbour ("island" — e.g. an LSOA separated
    from the rest of the study area) would otherwise be dropped from Moran's I,
    so islands are attached to their single nearest neighbour by centroid
    distance. That keeps every LSOA in the test.

    NOTE: the weights are positional — observation `i` of the weights matrix is
    row `i` of `gdf`. The caller must pass a frame in exactly the same row order
    as the values it later tests, and must not reorder it afterwards.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        LSOA polygons, in the row order that will be used for testing. Should
        be in a projected CRS (EPSG:27700) so that the island fallback measures
        centroid distance in metres.
    transform : str
        Weights transformation; "r" (row-standardised) is the usual choice and
        makes Moran's I directly interpretable.

    Returns
    -------
    libpysal.weights.W
    """
    if gdf.crs is not None and gdf.crs.to_epsg() != 27700:
        raise ValueError(
            f"Expected EPSG:27700 (British National Grid), got {gdf.crs.to_epsg()}. "
            "Reproject before building weights so centroid distances are in metres.")

    # use_index=False -> observations are identified by position (0..n-1)
    w = Queen.from_dataframe(gdf, use_index=False)

    if len(w.islands) > 0:
        if verbose:
            print(f"  {len(w.islands)} island(s) with no contiguous neighbour "
                  f"- attaching each to its nearest neighbour by centroid distance")
        w = attach_islands(w, KNN.from_dataframe(gdf, k=1))

    w.transform = transform

    if verbose:
        n_links = w.s0 if transform != "r" else sum(len(v) for v in w.neighbors.values())
        print(f"  Queen contiguity: {w.n} LSOAs, "
              f"mean {n_links / w.n:.1f} neighbours each")
    return w


# --------------------------------------------------------------------------
# Moran's I
# --------------------------------------------------------------------------

def morans(values, w, permutations=999, label=None):
    """Global Moran's I for one variable, with permutation-based inference.

    Moran's I is roughly a correlation coefficient between each value and the
    average of its neighbours' values: near +1 means strong clustering (like
    values next to each other), 0 means no spatial pattern, negative means a
    checkerboard.

    Significance is assessed by randomly reshuffling the values across space
    `permutations` times and seeing how often that produces an I as extreme as
    the observed one — so it makes no distributional assumptions.

    Parameters
    ----------
    values : array-like, shape (n_observations,)
        The variable to test — e.g. IMD rank, or a vector of out-of-fold
        residuals. Must be in the same row order as the frame `w` was built
        from, since the weights are positional.
    w : libpysal.weights.W
        Spatial weights, from `build_weights`.
    permutations : int
        Number of random reshuffles used to build the reference distribution.
        999 gives a smallest attainable pseudo p-value of 0.001, which is
        enough resolution for the tests here. Set to 0 to skip the permutations
        and fall back to the analytical normal approximation, which is faster
        but assumes normality — the returned `z` and `p` then come from that
        approximation instead.
    label : str or None
        Carried through to the returned dict unchanged, so that results for
        several variables can be collected into a DataFrame and still be told
        apart. Has no effect on the statistics.

    Returns
    -------
    dict with keys: label, I, E[I], z, p (pseudo p-value, one-sided)
    """
    values = np.asarray(values, dtype=float)
    mi = Moran(values, w, permutations=permutations)
    return {
        "label": label,
        "I": mi.I,
        "E[I]": mi.EI,
        "z": mi.z_sim if permutations else mi.z_norm,
        "p": mi.p_sim if permutations else mi.p_norm,
    }


def morans_many(matrix, w, permutations=99):
    """Run Moran's I on every column of a 2D array (e.g. all 512 embedding dims).

    Used to characterise the *distribution* of spatial autocorrelation across
    the predictors, rather than to make 512 individual claims. Fewer
    permutations are used than in `morans` because we only need the shape of
    the distribution, not a precise p-value for any single dimension (with 99
    permutations the smallest attainable pseudo p-value is 0.01).

    Parameters
    ----------
    matrix : array-like, shape (n_observations, n_variables)
        One column per variable to test. Rows must be in the same order as the
        observations `w` was built from.
    w : libpysal.weights.W
        Spatial weights, from `build_weights`.
    permutations : int
        Number of random reshuffles used for the pseudo p-value of each column.

    Returns
    -------
    pandas.DataFrame with one row per column of `matrix`.
    """
    matrix = np.asarray(matrix, dtype=float)
    rows = []
    for j in range(matrix.shape[1]):
        r = morans(matrix[:, j], w, permutations=permutations, label=j)
        r["dimension"] = j
        rows.append(r)
    return pd.DataFrame(rows).drop(columns="label")


# --------------------------------------------------------------------------
# Out-of-fold prediction and scoring
# --------------------------------------------------------------------------

def out_of_fold_predictions(model, X, y, cv, groups=None):
    """Predict every observation from the fold in which it was held out.

    Every LSOA appears in the test set of exactly one fold, so this yields one
    genuinely out-of-sample prediction per LSOA. That is what a residual
    autocorrelation test needs: residuals from a model that has *not* seen the
    LSOA in question. (Mapping in-sample residuals, as script 9b currently
    does, mixes fitted and predicted values and cannot support such a test.)

    Parameters
    ----------
    model : estimator
        Cloned before each fit, so the passed model is never mutated.
    cv : cross-validator
        e.g. KFold(shuffle=True) for the random split, or GroupKFold for the
        spatial one.
    groups : array-like, optional
        Required by GroupKFold; ignored by KFold.
    """
    y = np.asarray(y, dtype=float)
    y_pred = np.full(len(y), np.nan)

    for train_idx, test_idx in cv.split(X, y, groups):
        fold_model = clone(model)
        fold_model.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = fold_model.predict(X[test_idx])

    assert not np.isnan(y_pred).any(), "Some observations were never held out"
    return y_pred


def pooled_scores(y, y_pred):
    """Score all out-of-fold predictions together, against the global mean.

    Preferred over averaging per-fold R² when folds differ systematically. Under
    leave-one-LAD-out the folds are whole local authorities, whose deprivation
    distributions differ sharply (Manchester vs Trafford): a fold with a narrow
    spread of IMD ranks is scored against a small local variance and so gets a
    poor R² even from a good model. Pooling avoids that confound — the same
    reasoning that led script 8 to adopt NRMSE.
    """
    y = np.asarray(y, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    return {
        "R2": float(r2_score(y, y_pred)),
        "RMSE": rmse,
        "NRMSE": rmse / float(np.std(y)),
        "Spearman": float(spearmanr(y, y_pred).statistic),
    }


def per_fold_scores(y, X, cv, model, groups=None):
    """Fit per fold and return the per-fold R² / NRMSE, plus their means.

    Reported alongside `pooled_scores` because script 4's headline figure is a
    mean-of-folds R² (`GridSearchCV.best_score_`); this keeps the comparison
    like-for-like.
    """
    y = np.asarray(y, dtype=float)
    rows = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        fold_model = clone(model)
        fold_model.fit(X[train_idx], y[train_idx])
        y_pred = fold_model.predict(X[test_idx])
        rmse = float(np.sqrt(mean_squared_error(y[test_idx], y_pred)))
        rows.append({
            "fold": i,
            "held_out_group": (pd.unique(np.asarray(groups)[test_idx])[0]
                               if groups is not None else None),
            "n_test": len(test_idx),
            "R2": float(r2_score(y[test_idx], y_pred)),
            "RMSE": rmse,
            "NRMSE": rmse / float(np.std(y[test_idx])),
        })
    return pd.DataFrame(rows)
