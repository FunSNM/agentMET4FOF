import json
from pathlib import Path

import numpy as np
import pandas as pd
from time_series_metadata.scheme import MetaData

from agentMET4FOF.agents import MetrologicalAgent, MetrologicalMonitorAgent
from agentMET4FOF.network import AgentNetwork
from agentMET4FOF.sensornetwork.refurbed_classes import SensorPlatform
from agentMET4FOF.utils import Backend

quantity_unit_map = {'CO': '\mu g/m^3', 'NO': '\mu g/m^3', 'RH': 'percent', 'P': 'kPA', 'NO2': '\mu g/m^3',
                     'O3': '\mu g/m^3', 'PM10':'\mu g/m^3', 'PM25': '\mu g/m^3', 'T': '°C'}

# class OfficeRoomAgent(MetrologicalAgent):
#     """An agent representing a single room in a building streaming multiple signals corresponding
#     to the sensors placed inside the room.
#
#     Takes samples from a list of :py:mod:`Sensor` and pushes them sample by sample
#     to connected agents via its output channel.
#     """
#
#     def init_parameters(self, room_name: str = None, room_devices_dict: dict = None, room_data: pd.DataFrame = None, **kwargs):
#         """Initialize the input data
#
#         Initialize the input data stream(s) as an instances of the
#         :param room_name:
#         :param **kwargs:
#         :py:mod:`Sensor` class
#         """
#         self.sensor_platforms = []
#         self.room_name = room_name
#         # for key, val in room_devices_dict.items():
#         #     sensor_cols = [mystring for mystring in room_data.columns if (key in mystring or val[0] in mystring)]
#         for device, location  in room_devices_dict.items():
#             device_columns = [col for col in room_data.columns if device in col or any([loc in col for loc in location])]
#             device_data = room_data[device_columns]
#             self.sensor_platforms.append(SensorPlatform(platform_timeseries=device_data, platform_id=device))
#         super().init_parameters()
#
#     def agent_loop(self):
#         """Model the agent's behaviour
#
#         On state *Running* the agent will extract sample by sample the input data
#         streams content and push it via invoking :py:method:`AgentMET4FOF.send_output`.
#         """
#         if self.current_state == "Running":
#             # sine_data = self.sine_stream.next_sample()  # dictionary
#             # self.send_output(sine_data["quantities"])
#             for platform in self.sensor_platforms:
#                 platform.agent_loop()
#             super().agent_loop()
#
#     def n_sensors(self):
#         """Return the number of sensors in the room."""
#         return len(self.sensors)


class ConsensusAgent(MetrologicalAgent):
    ## Compute the consensus values of measurements

    def init_parameters(self, aggregate_keys=["O3"], consensus_function=np.nanmedian, floor_serialization=None, device_map=None):
        self.aggregate_keys = aggregate_keys
        self.consensus_function = consensus_function
        if type(consensus_function) == str:
            self.agg_function_name = consensus_function
        else:
            self.agg_function_name =  consensus_function.__name__

        self.floor_serialization = floor_serialization
        self.device_map = device_map
        self.connectivity_matrix = self.generate_connectivity_matrix()

        self.reputations = None  ## initialise sensor reputations

        self.aggregated_values = []
        self.metadata = MetaData(
            device_id="Consensus Agent " + " ".join(self.aggregate_keys),
            time_name='timestamp',
            time_unit='h',
            quantity_names=self.agg_function_name + " " + " ".join(self.aggregate_keys),
            quantity_units="\mu g/ m^3",
            misc="Consensus values "+" ".join(self.aggregate_keys)+ "in floor "+ f"{floor_serialization['floor']:02d}",
        )
        super().init_parameters()

    def on_received_message(self, message):
        #number of input agents that are connected to the aggregator
        n_input_agents = len(self.get_attr("Inputs"))

        if self.reputations is None:
            self.reputations = np.ones(n_input_agents)
        # store into buffer

        self.buffer.store(message["from"], {'data':message["data"], 'metadata':message["metadata"]})
        self.log_info(f"{self.buffer.values()}")

        ## compute mean of the latest value
        buffer_values = list(self.buffer.values())

        ## check 1: buffer len is the same
        if len(buffer_values) == n_input_agents:
            self.update_connectivity_matrix(buffer_values)
            self.reputations = self.update_reputations(buffer_values)
            self.buffer.clear()

    def generate_connectivity_matrix(self):

        connectivity_matrix = None
        sensor_platforms = []
        for room_id, devices in self.device_map.items():
            sensor_platforms_in_room = [device for device, range in devices.items() if 'airvolume' in range]
            n_airvolume_sensors = len(sensor_platforms_in_room)
            new_block = np.ones((n_airvolume_sensors, n_airvolume_sensors))
            if connectivity_matrix is None:
                connectivity_matrix = new_block
            else:
                zero_filler_matrix = np.zeros((connectivity_matrix.shape[0], n_airvolume_sensors))
                connectivity_matrix = np.block([[connectivity_matrix, zero_filler_matrix], [zero_filler_matrix.T, new_block]])
            sensor_platforms.extend(sensor_platforms_in_room)

        return pd.DataFrame(connectivity_matrix, index=sensor_platforms, columns=sensor_platforms)


    def update_connectivity_matrix(self,  buffer_values):
        # Load the JSON data

        door_states = {}
        #gather door states: 0 if closed, 2 if open
        for buffer_value in buffer_values:
            quant_and_platform_names = [(mdata._metadata['device_id'], mdata._metadata['misc'])  for mdata in buffer_value['metadata']]
            doornames = [qp[1].split(' ')[1]+'_'+qp[0][:5]
                         for qp in quant_and_platform_names if 'door' in qp[0]]
            door_indices = [i for i, s in enumerate(quant_and_platform_names) if "door" in s[0]]
            door_states.update({dname:buffer_value['data'][ind][2].astype(int) for dname, ind in zip(doornames, door_indices)})

        # If the rooms are nested under a key (e.g., data['rooms']), adjust accordingly
        rooms = self.floor_serialization if isinstance(self.floor_serialization, list) else self.floor_serialization.get('zones', [])

        # 1. Get a list of all unique room IDs to define the matrix dimensions
        room_ids = sorted([room['guid'] for room in rooms])
        id_to_idx = {room_id: i for i, room_id in enumerate(room_ids)}
        num_rooms = len(room_ids)

        matrix = np.zeros((num_rooms, num_rooms))

        for room in rooms:
            u_id = room['guid']
            sensor_platforms_in_room = [device for device, range in self.device_map[u_id].items() if
                                        'airvolume' in range]
            for wall in room.get('walls', []):
                for neighbour in wall.get('neighbours', []):
                    v_id = neighbour.get('space_id')
                    sensor_platforms_in_neighbour = [device for device, range in self.device_map[v_id].items() if
                                                'airvolume' in range]

                    if v_id in id_to_idx:
                        # Get the list of doors in this shared wall section [1]
                        doors = neighbour.get('door_ids_in_section', [])

                        if not doors:
                            # No door exists, only a shared wall
                            weight = 0.5
                        else:
                            # Door(s) exist. Check the state of the doors.
                            # If ANY door in the section is open (state != 2), weight is 1.
                            # If ALL doors in the section are closed (state == 2), weight is 0.5.
                            is_any_door_open = any(door_states.get(d_id, 0) != 2 for d_id in doors)
                            weight = 1.0 if is_any_door_open else 0.5

                        self.connectivity_matrix.loc[sensor_platforms_in_room,sensor_platforms_in_neighbour] = weight
                        self.connectivity_matrix.loc[sensor_platforms_in_neighbour,sensor_platforms_in_room] = weight


    def update_reputations(self, buffer_values):
        reputation_weights = self.reputations

def building_network_demonstration():
    agent_network = AgentNetwork(dashboard_modules=True, backend=Backend.MESA)
    sensor_metadata = pd.read_csv('Data/sensor_metadata.txt', sep='\t')
    #%%
    device_maps = {}
    device_folder = Path('Data/device_maps' )
    for file in device_folder.glob('*.json'):
        with file.open('r', encoding='utf-8') as f:
         device_maps[file.stem[12:]]=json.load(f)
    #%%
    building_ser = {}
    serialization_folder = Path('Data/building_serialization')
    for file in serialization_folder.glob('*.json'):
        with file.open('r', encoding='utf-8') as f:
            building_ser[file.stem[4:7]]=json.load(f)

    devices_by_room_and_floor = {floor: {room: device_maps[room] for room in device_maps if
                                         room in [zone['guid'] for zone in building_ser[floor]['zones']]} for floor in
                                 building_ser}
    #%%
    json.dump(devices_by_room_and_floor, open('Data/devices_by_room_and_floor.json', 'w'))
    json.dump(building_ser, open('Data/building_ser.json', 'w'))

    season_dict = {'20260707_20260713': 'summer', '20260112_20260119': 'winter'}
    room_timeseries = {}
    timeseries_folder = Path('Data/timeseries')
    for key in device_maps.keys():
        df_dict_internal = {}
        for file in timeseries_folder.glob(key[-7:]+'*'):
            df = pd.read_csv(file)
            basename=file.stem

            # pattern = r"(.*?)__(.*?)_calibration_offset__(.*)"
            # replacement = r"\1__\2__\3__calibration_offset"
            #
            # df.columns = [re.sub(pattern, replacement, col) for col in df.columns]
            offset_cols = [col for col in df.columns if 'calibration_offset' in col]

            for offset_col in offset_cols:
                # 2. Create the measurement column name by removing the offset string.
                # We include the underscore '_calibration_offset' so we don't leave a trailing underscore behind.
                meas_col = offset_col.replace('_calibration_offset', '')

                if meas_col in df.columns:
                    # Add the values safely
                    df[meas_col] = df[meas_col].add(df[offset_col], fill_value=0)
                else:
                    print(f"Warning: Could not find matching measurement column for {offset_col}")

                    # 2. Drop B
            df.drop(columns=offset_cols, inplace=True)

            df_dict_internal[season_dict[basename[9:26]]] = df.set_index('timestamp')
        room_timeseries[key] = df_dict_internal
    #%%
    floor_agents = {}
    monitor_agents = {}
    platform_agent_dict = {}
    for floor in building_ser:
        floor_agents[floor] = agent_network.add_agent(name="Consensus Agent " + floor, agentType=ConsensusAgent,
                                                      aggregate_keys='temperature',
                                                      floor_serialization=building_ser[floor], device_map=devices_by_room_and_floor[floor],
                                                      buffer_size=20)
        for zone in building_ser[floor]['zones']:
            rname = zone['guid']
            try:
                sdict = device_maps[rname]
                room_data = room_timeseries[rname]['winter']
                platform_agent_dict[rname] = []
                for device, location in sdict.items():
                    device_columns = [col for col in room_data.columns if
                                      device in col or any([loc in col for loc in location])]
                    if device_columns:
                        device_data = room_data[device_columns]

                        platform_agent = agent_network.add_agent(name="Sensor Platform " + device, agentType=SensorPlatform,
                                                                 platform_timeseries=device_data, position=rname,
                                                                 platform_id=device)

                        platform_agent.bind_output(floor_agents[floor])
                        platform_agent_dict[rname].append(platform_agent)
            except KeyError:
                print(f"Warning: Could not find sensor platforms for {rname}")


        monitor_agents[floor] = agent_network.add_agent(name="Monitor Agent " + floor,
                                                        agentType=MetrologicalMonitorAgent)

        floor_agents[floor].bind_output(monitor_agents[floor])

    agent_network.set_running_state()
    return agent_network

if __name__ == '__main__':
    building_network_demonstration()