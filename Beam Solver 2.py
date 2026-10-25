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
        [1_800_000, 0.024, 13_866.20719, -6_800.0, 2_493.636485, 1_190.490678,],  #Oak
        [1_500_000, 0.014, 10_573.61133, -5_000.0, 1_439.664333, 642.9137631,],   #Pine
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

@njit
def evaluate_I_beam(
    bottom_height,  bottom_width,   bottom_material_ID,
    web_height,     web_width,      web_material_ID,
    top_height,     top_width,      top_material_ID,
):
    #Convert to inches
    bottom_height /= 16.0
    bottom_width /= 16.0
    web_height /= 16.0
    web_width /= 16.0
    top_height /= 16.0
    top_width /= 16.0
    
    weight = (
        bottom_height * bottom_width * MAT_PROPS[bottom_material_ID, Prop.DENSITY] + 
        web_height * web_width * MAT_PROPS[web_material_ID, Prop.DENSITY] +
        top_height * top_width * MAT_PROPS[top_material_ID, Prop.DENSITY]
    ) * BEAM_LENGTH
    
    #Calculate transformed areas
    bottom_area_E = bottom_height * bottom_width * MAT_PROPS[bottom_material_ID, Prop.E]
    web_area_E = web_height * web_width * MAT_PROPS[web_material_ID, Prop.E]
    top_area_E = top_height * top_width * MAT_PROPS[top_material_ID, Prop.E]
    
    #Calculate height of centroid of each section
    bottom_y = bottom_height / 2.0
    web_y = bottom_height + web_height / 2.0
    top_y = bottom_height + web_height + top_height / 2.0
    
    #Calculate y_bar
    y_bar = (bottom_area_E * bottom_y + web_area_E * web_y + top_area_E * top_y) / (bottom_area_E + web_area_E + top_area_E)
    
    if not(bottom_height < y_bar < bottom_height + web_height):
        #print("this beam probably sucks + I'm too lazy to add if statements for y_bar in other places")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    #Calculate I
    bottom_I_E = bottom_area_E * (1.0 / 12.0 * bottom_height ** 2 + (bottom_y - y_bar) ** 2)
    web_I_E = web_area_E * (1.0 / 12.0 * web_height ** 2 + (web_y - y_bar) ** 2)
    top_I_E = top_area_E * (1.0 / 12.0 * top_height ** 2 + (top_y - y_bar) ** 2)
    total_I_E = bottom_I_E + web_I_E + top_I_E
    
    #Calculate force limits from bending stress
    #F=(-sigma*I/y-M_COEFF_W*weight)/M_COEFF_F
    f_tension_bottom = ((-MAT_PROPS[bottom_material_ID, Prop.TENSILE] * total_I_E / MAT_PROPS[bottom_material_ID, Prop.E] / (0 - y_bar)) - M_COEFF_W * weight) / M_COEFF_F
    f_tension_web = ((-MAT_PROPS[web_material_ID, Prop.TENSILE] * total_I_E / MAT_PROPS[web_material_ID, Prop.E] / (bottom_height - y_bar)) - M_COEFF_W * weight) / M_COEFF_F
    
    f_compression_web = ((-MAT_PROPS[web_material_ID, Prop.COMPRESSIVE] * total_I_E / MAT_PROPS[web_material_ID, Prop.E] / (bottom_height + web_height - y_bar)) - M_COEFF_W * weight) / M_COEFF_F
    f_compression_top = ((-MAT_PROPS[top_material_ID, Prop.COMPRESSIVE] * total_I_E / MAT_PROPS[top_material_ID, Prop.E] / (bottom_height + web_height + top_height - y_bar)) - M_COEFF_W * weight) / M_COEFF_F
    
    #Calculate force limits from glue shear
    #F=(tau * I * thickness / Q - V_COEFF_W * weight) / V_COEFF_F
    #TODO: Check with prof what to use for I, t, and Q
    bottom_min_glue_shear = min(MAT_PROPS[bottom_material_ID, Prop.GLUE_SHEAR], MAT_PROPS[web_material_ID, Prop.GLUE_SHEAR])
    top_min_glue_shear = min(MAT_PROPS[web_material_ID, Prop.GLUE_SHEAR], MAT_PROPS[top_material_ID, Prop.GLUE_SHEAR])
    
    bottom_Q_E = bottom_area_E * (y_bar - bottom_y)
    top_Q_E = top_area_E * (top_y - y_bar)
    
    f_glue_bottom = (bottom_min_glue_shear * total_I_E * web_width / bottom_Q_E - V_COEFF_W * weight) / V_COEFF_F
    f_glue_top = (top_min_glue_shear * total_I_E * web_width / top_Q_E - V_COEFF_W * weight) / V_COEFF_F
    
    #Calculate force limit from web shear
    #F=(tau * I * thickness / Q - V_COEFF_W * weight) / V_COEFF_F
    web_shear = MAT_PROPS[web_material_ID, Prop.SHEAR]
    
    web_Q_E = (y_bar - bottom_height) * web_width * MAT_PROPS[web_material_ID, Prop.E] * (y_bar - bottom_height) / 2
    #Define it again for clarity or smth idk
    bottom_Q_E = bottom_area_E * (y_bar - bottom_y)
    
    centroid_Q_E = web_Q_E + bottom_Q_E
    
    f_web_shear = (web_shear * total_I_E * web_width / centroid_Q_E - V_COEFF_W * weight) / V_COEFF_F
    
    f_tension = min(f_tension_bottom, f_tension_web)
    f_compression = min(f_compression_web, f_compression_top)
    f_glue_shear = min(f_glue_bottom, f_glue_top)
    
    f_fail = min(f_tension, f_compression, f_glue_shear, f_web_shear, MAX_FORCE)
    
    if f_fail < MIN_FORCE:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    efficiency = f_fail / weight
    
    return efficiency, f_fail, weight, f_tension, f_compression, f_glue_shear, f_web_shear

@njit
def _run_single_task(task_id):
    local_max = 0.0
    local_count = 0
    
    bottom_height = 3 + task_id // 10
    top_height = 3 + task_id % 10
    
    max_bottom_width = min(2 * 16, bottom_height * 8)
    max_top_width = min(2 * 16, top_height * 8)
    
    for web_width in range(3, 12 + 1):
        for bottom_width in range(web_width + 1, max_bottom_width + 1):
            for top_width in range(web_width + 1, max_top_width + 1):
                max_width = max(bottom_width, top_width)
                max_web_height = min((2 * max_width) - bottom_height - top_height, web_width * 8)
                
                if max_web_height < 3:
                    continue
                
                for web_height in range(3, max_web_height + 1):
                    for n in range(2 ** 3):
                        bottom_material_ID = (n >> 2) & 0b1
                        web_material_ID = (n >> 1) & 0b1
                        top_material_ID = (n >> 0) & 0b1
                        
                        efficiency = evaluate_I_beam(
                            bottom_height,  bottom_width,   bottom_material_ID,
                            web_height,     web_width,      web_material_ID,
                            top_height,     top_width,      top_material_ID
                        )[0]
                        
                        local_count += 1
                        
                        if efficiency > local_max:
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
    
    bottom_height = 3 + task_id // 10
    top_height = 3 + task_id % 10
    
    max_bottom_width = min(2 * 16, bottom_height * 8)
    max_top_width = min(2 * 16, top_height * 8)
    
    for web_width in range(3, 12 + 1):
        for bottom_width in range(web_width + 1, max_bottom_width + 1):
            for top_width in range(web_width + 1, max_top_width + 1):
                max_width = max(bottom_width, top_width)
                max_web_height = min((2 * max_width) - bottom_height - top_height, web_width * 8)
                
                if max_web_height < 3:
                    continue
                
                for web_height in range(3, max_web_height + 1):
                    for n in range(2 ** 3):
                        bottom_material_ID = (n >> 2) & 0b1
                        web_material_ID = (n >> 1) & 0b1
                        top_material_ID = (n >> 0) & 0b1
                        
                        efficiency, f_fail, weight, f_tension, f_compression, f_glue_shear, f_web_shear = evaluate_I_beam(
                            bottom_height,  bottom_width,   bottom_material_ID,
                            web_height,     web_width,      web_material_ID,
                            top_height,     top_width,      top_material_ID
                        )
                        
                        if efficiency >= threshold and local_count < max_buffer_per_task:
                            b = task_buffer[local_count]
                            b[0] = efficiency
                            b[1] = f_fail
                            b[2] = weight
                            b[3] = f_tension
                            b[4] = f_compression
                            b[5] = f_glue_shear
                            b[6] = f_web_shear
                            b[7] = bottom_height
                            b[8] = bottom_width
                            b[9] = bottom_material_ID
                            b[10] = web_height
                            b[11] = web_width
                            b[12] = web_material_ID
                            b[13] = top_height
                            b[14] = top_width
                            b[15] = top_material_ID
                            
                            local_count += 1
                        elif local_count >= max_buffer_per_task:
                            is_full = True
                            
    return local_count, is_full

@njit(parallel=True)
def collect_best_designs(threshold, max_buffer_per_task=2000):
    NUM_TASKS = 100
    buffer = np.zeros((NUM_TASKS, max_buffer_per_task, 16), dtype=np.float64)
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
        "Bottom Height (/16 in)", "Bottom Width (/16 in)", "Bottom Material",
        "Web Height (/16 in)", "Web Width (/16 in)", "Web Material",
        "Top Height (/16 in)", "Top Width (/16 in)", "Top Material"
    ]
    df = pd.DataFrame(data, columns=columns)

    # Export to CSV
    csv_filename = "beam_optimization_results.csv"
    print(f"\n4. Saving to {csv_filename}")
    
    df.to_csv(csv_filename, index=False)
    print(f"Done!")