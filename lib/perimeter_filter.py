"""
Adaptive keep-buffer perimeter filtering (shared by create_csv_prediction and runPrediction).
Same logic as compareModelAccuracy / view_perims_jupyter.
"""
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt


def analyze_perimeter_changes(
    current_mask,
    previous_mask,
    alignment_buffer_pixels=7,
    vulnerable_radius=50,
    keep_buffer_pixels=15,
    adaptive_keep_buffer=True,
    buffer_scale=3,
    min_buffer_radius=5,
    max_buffer_radius=200,
):
    """
    Compute filter_zones: pixels to exclude from metrics (vulnerable zone minus keep buffer and stable core).
    When adaptive_keep_buffer=True, effective keep radius = clamp(keep_buffer_pixels * buffer_scale, min_buffer_radius, max_buffer_radius).
    Returns: (true_growth, stable_core, filter_zones, alignment_adjusted_overlap, raw_regression).
    """
    if current_mask is None or previous_mask is None:
        return None, None, None, None, None

    current_mask = np.asarray(current_mask).astype(bool)
    previous_mask = np.asarray(previous_mask).astype(bool)
    if current_mask.ndim > 2:
        current_mask = current_mask[:, :, 0]
    if previous_mask.ndim > 2:
        previous_mask = previous_mask[:, :, 0]

    if alignment_buffer_pixels > 0:
        current_buffered = binary_dilation(current_mask, iterations=alignment_buffer_pixels)
        previous_buffered = binary_dilation(previous_mask, iterations=alignment_buffer_pixels)
    else:
        current_buffered = current_mask
        previous_buffered = previous_mask
    raw_growth = current_mask & ~previous_mask
    raw_regression = previous_mask & ~current_mask
    raw_overlap = current_mask & previous_mask
    alignment_adjusted_overlap = (current_mask & previous_buffered) | (previous_mask & current_buffered)
    true_growth = current_mask & ~previous_buffered
    stable_core = raw_overlap
    non_growth_areas = current_mask & ~true_growth

    if vulnerable_radius > 0:
        distance_from_non_growth = distance_transform_edt(~non_growth_areas)
        vulnerable_zone = distance_from_non_growth <= vulnerable_radius
        print(f"Using Euclidean distance transform for vulnerable radius {vulnerable_radius}")
    else:
        vulnerable_zone = non_growth_areas

    if adaptive_keep_buffer:
        effective_keep_radius = max(
            min_buffer_radius,
            min(max_buffer_radius, int(keep_buffer_pixels * buffer_scale)),
        )
        print(f"Adaptive keep buffer: {keep_buffer_pixels} * {buffer_scale} -> effective radius {effective_keep_radius} (clamp [{min_buffer_radius}, {max_buffer_radius}])")
    else:
        effective_keep_radius = keep_buffer_pixels

    if effective_keep_radius > 0 and true_growth is not None:
        distance_from_true_growth = distance_transform_edt(~true_growth)
        keep_buffer_zone = distance_from_true_growth <= effective_keep_radius
        print(f"Using Euclidean distance transform for keep buffer {effective_keep_radius} px")
    else:
        keep_buffer_zone = true_growth if true_growth is not None else np.zeros_like(current_mask, dtype=bool)

    filter_zones = vulnerable_zone.copy()
    if keep_buffer_zone is not None:
        filter_zones = filter_zones & ~keep_buffer_zone
    if stable_core is not None:
        filter_zones = filter_zones & ~stable_core

    total_pixels = current_mask.size
    current_total = np.sum(current_mask)
    previous_total = np.sum(previous_mask) if previous_mask is not None else 0
    true_growth_pixels = np.sum(true_growth) if true_growth is not None else 0
    stable_core_pixels = np.sum(stable_core) if stable_core is not None else 0
    raw_growth_pixels = np.sum(raw_growth) if raw_growth is not None else 0
    raw_regression_pixels = np.sum(raw_regression) if raw_regression is not None else 0
    alignment_overlap_pixels = np.sum(alignment_adjusted_overlap) if alignment_adjusted_overlap is not None else 0
    non_growth_pixels = np.sum(non_growth_areas) if non_growth_areas is not None else 0
    vulnerable_pixels = np.sum(vulnerable_zone) if vulnerable_zone is not None else 0
    keep_buffer_pixels_count = np.sum(keep_buffer_zone) if keep_buffer_zone is not None else 0
    filter_pixels = np.sum(filter_zones) if filter_zones is not None else 0
    keep_pixels = true_growth_pixels + stable_core_pixels

    print(f"\nPerimeter Analysis (alignment buffer: {alignment_buffer_pixels}px, vulnerable radius: {vulnerable_radius}px, keep buffer: {effective_keep_radius}px):")
    print(f"  Current fire total:           {current_total:6d} pixels ({100*current_total/total_pixels:5.1f}%)")
    print(f"  Previous fire total:         {previous_total:6d} pixels ({100*previous_total/total_pixels:5.1f}%)")
    print(f"  Raw growth (before align):   {raw_growth_pixels:6d} pixels ({100*raw_growth_pixels/total_pixels:5.1f}%)")
    print(f"  Raw regression:              {raw_regression_pixels:6d} pixels ({100*raw_regression_pixels/total_pixels:5.1f}%)")
    print(f"  Raw overlap:                 {stable_core_pixels:6d} pixels ({100*stable_core_pixels/total_pixels:5.1f}%)")
    print(f"  Alignment-adjusted overlap:  {alignment_overlap_pixels:6d} pixels ({100*alignment_overlap_pixels/total_pixels:5.1f}%)")
    print(f"  TRUE GROWTH (KEEP):          {true_growth_pixels:6d} pixels ({100*true_growth_pixels/total_pixels:5.1f}%)")
    print(f"  Stable core (KEEP):          {stable_core_pixels:6d} pixels ({100*stable_core_pixels/total_pixels:5.1f}%)")
    print(f"  Non-growth areas:           {non_growth_pixels:6d} pixels ({100*non_growth_pixels/total_pixels:5.1f}%)")
    print(f"  Vulnerable zone:             {vulnerable_pixels:6d} pixels ({100*vulnerable_pixels/total_pixels:5.1f}%)")
    print(f"  Keep buffer zone:            {keep_buffer_pixels_count:6d} pixels ({100*keep_buffer_pixels_count/total_pixels:5.1f}%)")
    print(f"  FILTER ZONES (final):        {filter_pixels:6d} pixels ({100*filter_pixels/total_pixels:5.1f}%)")
    print(f"  TOTAL KEEP:                  {keep_pixels:6d} pixels ({100*keep_pixels/total_pixels:5.1f}%)")
    print(f"  Misalignment correction:    {raw_growth_pixels - true_growth_pixels:6d} pixels corrected")
    print(f"  Vulnerable areas protected: {vulnerable_pixels - filter_pixels:6d} pixels protected by keep buffer")

    return true_growth, stable_core, filter_zones, alignment_adjusted_overlap, raw_regression
