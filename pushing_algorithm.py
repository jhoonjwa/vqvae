import json
import numpy as np
from tqdm import tqdm

def return_array(dataset, grid_size=512):
    array_data = dataset["detailed_grid_counts"]
    grid = np.zeros((grid_size, grid_size), dtype=int)    
    
    for key, value in array_data.items():
        x, y = map(int, key.split(','))
        grid[x-1, y-1] += int(value)
    
    return grid

def return_direction(grid, i, j):
    rows, cols = len(grid), len(grid[0])
    
    # Count zeros in each region
    right = sum(grid[i][x] == 0 for x in range(j+1, cols))  # x > i (same row), y > j
    left = sum(grid[i][x] == 0 for x in range(j))           # x = i (same row), y < j  
    down = sum(grid[x][j] == 0 for x in range(i+1, rows))   # x > i, y = j (same col)
    up = sum(grid[x][j] == 0 for x in range(i))             # x < i, y = j (same col)
    
    # Find direction with most zeros
    counts = [right, down, left, up]
    max_idx = counts.index(max(counts))
    
    return [(1, 0), (0, 1), (-1, 0), (0, -1)][max_idx]

def flatten_vertices(grid, i, j, directions):
    d1, d2 = directions
    rows, cols = grid.shape
    
    if grid[i, j] <= 1:
        return
    
    excess = grid[i, j] - 1
    grid[i, j] = 1
    
    failed = 0
    
    # Determine direction to push
    if d1 != 0:  # Horizontal movement
        step = 1 if d1 > 0 else -1
        start_col = j + step
        end_col = cols if d1 > 0 else -1
        
        current_col = start_col
        remaining = excess
        
        while remaining > 0 and current_col != end_col:
            if grid[i, current_col] == 0:
                grid[i, current_col] = 1
                remaining -= 1
            else:
                # Push existing element to next position
                existing_value = grid[i, current_col]
                grid[i, current_col] = 1
                remaining -= 1
                
                # Push existing value forward - replace or chain push
                next_col = current_col + step
                while existing_value > 0 and next_col != end_col:
                    if grid[i, next_col] == 0:
                        grid[i, next_col] = existing_value
                        existing_value = 0
                    else:
                        # Chain push - swap values
                        temp = grid[i, next_col]
                        grid[i, next_col] = existing_value
                        existing_value = temp
                        next_col += step
                    if next_col == end_col:
                        failed += 1

            current_col += step
            
    else:  # Vertical movement (d2 != 0)
        step = 1 if d2 > 0 else -1
        start_row = i + step
        end_row = rows if d2 > 0 else -1
        
        current_row = start_row
        remaining = excess
        
        while remaining > 0 and current_row != end_row:
            if grid[current_row, j] == 0:
                grid[current_row, j] = 1
                remaining -= 1
            else:
                # Push existing element to next position
                existing_value = grid[current_row, j]
                grid[current_row, j] = 1
                remaining -= 1
                
                # Push existing value forward - replace or chain push
                next_row = current_row + step
                while existing_value > 0 and next_row != end_row:
                    if grid[next_row, j] == 0:
                        grid[next_row, j] = existing_value
                        existing_value = 0
                    else:
                        # Chain push - swap values
                        temp = grid[next_row, j]
                        grid[next_row, j] = existing_value
                        existing_value = temp
                        next_row += step
                        if next_row == end_row:
                            failed += 1
            current_row += step
    if failed > 0:
        print(f"Failed pushes: {failed}")

def pushing_algorithm(grid, duplicate_limit=1):
    for i in tqdm(range(grid.shape[0])):
        for j in range(grid.shape[1]):
            if grid[i, j] > duplicate_limit:
                d1, d2 = return_direction(grid, i, j)
                flatten_vertices(grid, i, j, (d1, d2)) 
    
    return grid

def count_excess(grid, duplicate_limit=1):
    excess = 0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] > duplicate_limit:
                excess += 1
                print(f"Excess at ({i}, {j}): {grid[i, j]}")
    return excess


def main():
    path = r"C:\Users\user\Desktop\ALIN\3D LLM Agent\push_algorithm_test\0bc1dbec12a944f2b0bde183c67057c1_spherical_projection.json"
    with open(path, 'r') as file:
        dataset = json.load(file)
    
    grid = return_array(dataset)
    
    grid_sum = np.sum(grid)
    gt_sum = len(dataset["detailed_grid_counts"].keys())
    print(f"Grid sum: {grid_sum}, GT sum: {gt_sum}")
    excess = count_excess(grid)
    print(f"Excess elements before pushing: {excess}")
    final_grid = pushing_algorithm(grid)    
    excess_after = count_excess(final_grid)
    print(f"Excess elements after pushing: {excess_after}")
    

if __name__ == "__main__":
    main()