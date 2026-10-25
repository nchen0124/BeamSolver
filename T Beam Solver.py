import time
from enum import IntEnum
import numpy as np
from numba import njit, prange
import pandas as pd

THRESHOLD_RATIO = 0.9

class Prop(IntEnum):
  E = 0             #Elastic Modulus (psi)
  DENSITY = 1       #Density (lb/in^3)
  TENSILE = 2       #Tensile Strength (psi)
  COMPRESSIVE = 3   #Compressive Strength (psi)
  SHEAR = 4         #Shear Strength (psi)
  GLUE_SHEAR = 5

#TODO: get better numbers
MAT_PROPS = np.array(
    [
        [1_800_000, 0.024, 17872.82353, -6_800.0, 2872.4, 1391.1,],     #Oak
        [1_500_000, 0.014, 14326.58824, -5_000.0, 1491.75, 988.8,],     #Pine
    ],
    dtype=np.float64,
)

BEAM_LENGTH = 22.0
MIN_FORCE = 1000.0
MAX_FORCE = 2500.0
V_COEFF_F = 0.6
M_COEFF_F = 4.8
V_COEFF_W = 5.0 / 11.0
M_COEFF_W = 95.0 / 44.0
OVERDESIGN_RATIO = 1.2

@njit
def evaluate_T_beam(
    web_height,     web_width,      web_material_ID,
    top_height,     top_width,      top_material_ID,
):
    # Convert from 1/16ths to inches
    web_height /= 16.0
    web_width /= 16.0
    top_height /= 16.0
    top_width /= 16.0
    
    weight = (
        web_height * web_width * MAT_PROPS[web_material_ID, Prop.DENSITY] +
        top_height * top_width * MAT_PROPS[top_material_ID, Prop.DENSITY]
    ) * BEAM_LENGTH
    
    # Calculate transformed areas (E * A)
    web_area_E = web_height * web_width * MAT_PROPS[web_material_ID, Prop.E]
    top_area_E = top_height * top_width * MAT_PROPS[top_material_ID, Prop.E]
    
    # Section centroids from bottom
    web_y = web_height / 2.0
    top_y = web_height + top_height / 2.0
    
    # Composite neutral axis from bottom
    y_bar = (web_area_E * web_y + top_area_E * top_y) / (web_area_E + top_area_E)
    
    # Transformed moment of inertia (E * I)
    web_I_E = web_area_E * (1.0 / 12.0 * web_height ** 2 + (web_y - y_bar) ** 2)
    top_I_E = top_area_E * (1.0 / 12.0 * top_height ** 2 + (top_y - y_bar) ** 2)
    total_I_E = web_I_E + top_I_E
    
    f_tension = ((MAT_PROPS[web_material_ID, Prop.TENSILE] * total_I_E / MAT_PROPS[web_material_ID, Prop.E] / y_bar) - M_COEFF_W * weight) / M_COEFF_F
    
    f_compression_top = ((-MAT_PROPS[top_material_ID, Prop.COMPRESSIVE] * total_I_E / MAT_PROPS[top_material_ID, Prop.E] / (web_height + top_height - y_bar)) - M_COEFF_W * weight) / M_COEFF_F
    
    top_Q_E = top_area_E * (top_y - y_bar)
    top_min_glue_shear = min(MAT_PROPS[web_material_ID, Prop.GLUE_SHEAR], MAT_PROPS[top_material_ID, Prop.GLUE_SHEAR])
    f_glue_shear = (top_min_glue_shear * total_I_E * web_width / top_Q_E - V_COEFF_W * weight) / V_COEFF_F

    if y_bar < web_height:
        f_compression_web = ((-MAT_PROPS[web_material_ID, Prop.COMPRESSIVE] * total_I_E / MAT_PROPS[web_material_ID, Prop.E] / (web_height - y_bar)) - M_COEFF_W * weight) / M_COEFF_F
        f_compression = min(f_compression_web, f_compression_top)
        
        centroid_Q_E = web_width * y_bar * MAT_PROPS[web_material_ID, Prop.E] * (y_bar / 2.0)
    else:
        f_compression = f_compression_top
        
        centroid_Q_E = top_Q_E
        
    web_shear = MAT_PROPS[web_material_ID, Prop.SHEAR]
    f_web_shear = (web_shear * total_I_E * web_width / centroid_Q_E - V_COEFF_W * weight) / V_COEFF_F
    
    f_fail = min(f_tension, f_compression, f_glue_shear, f_web_shear, MAX_FORCE)
    
    if f_fail <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
    efficiency = f_fail / weight
    return efficiency, f_fail, weight, f_tension, f_compression, f_glue_shear, f_web_shear

@njit
def _run_single_task(task_id):
    local_max = 0.0
    local_count = 0
    
    web_width = 3 + task_id // 10
    top_height = 3 + task_id % 10
    
    max_top_width = min(2 * 16, top_height * 8)
    
    for top_width in range(web_width + 1, max_top_width + 1):
        max_web_height = min((2 * top_width) - top_height, web_width * 8)
        
        if max_web_height < 3:
            continue
        
        for web_height in range(3, max_web_height + 1):
            for n in range(2 ** 2):
                web_material_ID = (n >> 1) & 0b1
                top_material_ID = (n >> 0) & 0b1
                
                efficiency, f_fail, weight, f_tension, f_compression, f_glue_shear, f_web_shear = evaluate_T_beam(
                    web_height,     web_width,      web_material_ID,
                    top_height,     top_width,      top_material_ID
                )
                
                local_count += 1
                
                if efficiency > local_max:
                    if f_compression / f_fail <= OVERDESIGN_RATIO and f_glue_shear / f_fail <= OVERDESIGN_RATIO and f_web_shear / f_fail <= OVERDESIGN_RATIO:
                        local_max = efficiency
                            
    return local_max, local_count

@njit(parallel=True)
def find_max_efficiency():
    NUM_TASKS = 100
    task_max_efficiency = np.zeros(NUM_TASKS, dtype=np.float64)
    counts = np.zeros(NUM_TASKS, dtype=np.int64)

    for task_id in prange(NUM_TASKS):
        local_max, local_count = _run_single_task(task_id)
        task_max_efficiency[task_id] = local_max
        counts[task_id] = local_count

    return np.max(task_max_efficiency), np.sum(counts)

@njit
def _collect_single_task(task_id, threshold, task_buffer, max_buffer_per_task):
    local_count = 0
    is_full = False
    
    web_width = 3 + task_id // 10
    top_height = 3 + task_id % 10
    
    max_top_width = min(2 * 16, top_height * 8)
    
    for top_width in range(web_width + 1, max_top_width + 1):
        max_web_height = min((2 * top_width) - top_height, web_width * 8)
        
        if max_web_height < 3:
            continue
        
        for web_height in range(3, max_web_height + 1):
            for n in range(2 ** 2):
                web_material_ID = (n >> 1) & 0b1
                top_material_ID = (n >> 0) & 0b1
                
                efficiency, f_fail, weight, f_tension, f_compression, f_glue_shear, f_web_shear = evaluate_T_beam(
                    web_height,     web_width,      web_material_ID,
                    top_height,     top_width,      top_material_ID
                )
                        
                if efficiency >= threshold and local_count < max_buffer_per_task:
                    if f_compression / f_fail <= OVERDESIGN_RATIO and f_glue_shear / f_fail <= OVERDESIGN_RATIO and f_web_shear / f_fail <= OVERDESIGN_RATIO:
                        b = task_buffer[local_count]
                        b[0] = efficiency
                        b[1] = f_fail
                        b[2] = weight
                        b[3] = f_tension
                        b[4] = f_compression
                        b[5] = f_glue_shear
                        b[6] = f_web_shear
                        b[7] = web_height
                        b[8] = web_width
                        b[9] = web_material_ID
                        b[10] = top_height
                        b[11] = top_width
                        b[12] = top_material_ID
                        
                        local_count += 1
                elif local_count >= max_buffer_per_task:
                    is_full = True

    return local_count, is_full

@njit(parallel=True)
def collect_best_designs(threshold, max_buffer_per_task=2000):
    NUM_TASKS = 100
    buffer = np.zeros((NUM_TASKS, max_buffer_per_task, 13), dtype=np.float64)
    full = np.zeros(NUM_TASKS, dtype=np.bool_)  
    counts = np.zeros(NUM_TASKS, dtype=np.int64)

    for task_id in prange(NUM_TASKS):
        local_count, is_full = _collect_single_task(task_id, threshold, buffer[task_id], max_buffer_per_task)
        counts[task_id] = local_count
        full[task_id] = is_full
        
    return buffer, counts, np.any(full)

if __name__ == "__main__":
    print("1. Finding Maximum Efficiency")
    
    t0 = time.perf_counter()
    max_efficiency, total_count = find_max_efficiency()
    t1 = time.perf_counter()
    
    print(f"Maximum Efficiency found: {max_efficiency:,.2f}lb/lb from {total_count:,} designs\nTook {t1-t0:.3f}s")
    
    threshold = max_efficiency * THRESHOLD_RATIO
    print(f"\n2. Collecting designs with efficiency >= {threshold:,.2f}")
    
    t2 = time.perf_counter()
    buffer, save_counts, full = collect_best_designs(threshold, 5000)
    t3 = time.perf_counter()
    
    print(f"{np.sum(save_counts):,} designs collected\nTook {t3-t2:.3f}s{f"\nWarning, buffers filled" if np.any(full) else ""}")
    
    print(f"\n3. Processing for export to csv")
    all_rows = []
    for task in range(100):
        if save_counts[task] > 0:
            all_rows.append(buffer[task, :save_counts[task]])
    
    data = np.vstack(all_rows)

    # Convert to DataFrame
    columns = [
        "Efficiency (lb/lb)", "Max_Force (lb)", "Weight (lb)", "Tension (lb)", "Compression (lb)", "Glue Shear (lb)", "Wood Shear (lb)",
        "Web Height (/16 in)", "Web Width (/16 in)", "Web Material",
        "Top Height (/16 in)", "Top Width (/16 in)", "Top Material"
    ]
    df = pd.DataFrame(data, columns=columns)

    # Export to CSV
    csv_filename = "beam_optimization_results_20_Percent.csv"
    print(f"\n4. Saving to {csv_filename}")
    
    df.to_csv(csv_filename, index=False)
    print(f"Done!")