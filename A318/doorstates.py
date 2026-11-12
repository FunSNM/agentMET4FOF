import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np

# df = pd.read_csv('Data/timeseries/B01_R17__20260707_20260713_excl__5min.csv')
#
# doorcols = [col for col in df.columns if "door" in col]
#
# for col in doorcols:
#     plt.plot(df[[col]])
#     print(df[[col]].sum())
#
# plt.show()


def generate_connectivity_matrix(json_file_path):
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Assuming the top-level structure is a list of room objects
    # If the rooms are nested under a key (e.g., data['rooms']), adjust accordingly
    rooms = data if isinstance(data, list) else data.get('zones', [])

    # 1. Get a list of all unique room IDs to define the matrix dimensions
    room_ids = sorted([room['guid'] for room in rooms])
    id_to_idx = {room_id: i for i, room_id in enumerate(room_ids)}
    num_rooms = len(room_ids)

    # Initialize the matrix with 0
    matrix = np.zeros((num_rooms, num_rooms))

    # 2. Iterate through each room to find connectivity
    for room in rooms:
        u_id = room['guid']
        u_idx = id_to_idx[u_id]

        # Check all walls of the current room
        for wall in room.get('walls', []):
            # Check all neighbours of the current wall
            for neighbour in wall.get('neighbours', []):
                v_id = neighbour.get('space_id')

                # Ensure the neighbor exists in our room list
                if v_id in id_to_idx:
                    v_idx = id_to_idx[v_id]

                    # Determine connectivity value:
                    # 1 if there is a door in the shared section, 0.5 otherwise
                    doors = neighbour.get('door_ids_in_section', [])
                    weight = 1.0 if len(doors) > 0 else 0.5

                    # Set value in the matrix (symmetrical)
                    matrix[u_idx][v_idx] = weight
                    matrix[v_idx][u_idx] = weight

    # Convert to DataFrame for better readability
    np.fill_diagonal(matrix, 1)
    connectivity_df = pd.DataFrame(matrix, index=room_ids, columns=room_ids)
    return connectivity_df

# Example usage:
df = generate_connectivity_matrix('Data/building_serialization/B01_F02_serialization.json')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)