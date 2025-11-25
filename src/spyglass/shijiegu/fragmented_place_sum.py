import numpy as np

def sum_place_field(normalized_placefields,cells):
    cell_num = 0
    placefields = []
    for cell in cells:
        (e,u) = cell
        tmp = normalized_placefields[(e,u)]
        if np.sum(~np.isnan(tmp)) != 0:
            placefields.append(normalized_placefields[(e,u)])
            cell_num = cell_num + 1
    placefields = np.array(placefields)
    placefields_sum = np.sum(placefields, axis = 0)
    return placefields_sum, cell_num

    
def sum_place_field_by_weight(normalized_placefields,cells,weights):
    cell_num = 0
    valid_index = []
    for cell_ind in range(len(cells)):
        cell = cells[cell_ind]
        tmp = normalized_placefields[tuple(cell)]
        if np.sum(~np.isnan(tmp)) != 0:
            valid_index.append(cell_ind)
            cell_num = cell_num + 1
        else:
            weights[cell_ind] = 0

    weights = weights/np.sum(weights)
    placefields_intvl_sum = np.sum(
        np.array(
            [normalized_placefields[tuple(cells[ind])] * weights[ind] for ind in valid_index]
        ),
        axis = 0)
    return placefields_intvl_sum, cell_num