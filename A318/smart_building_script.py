import itertools
import json
from collections import defaultdict
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

class ConsensusAgent(MetrologicalAgent):
    ## Compute the consensus values of measurements

    def init_parameters(self, floor_name=None, consensus_function=np.nanmedian, floor_serialization=None,
                        device_map=None, quantities=['temperature', 'humidity', 'co2'], update_rate = 0.1):
        self.floor_name = floor_name
        self.consensus_function = consensus_function
        if type(consensus_function) == str:
            self.consensus_function_name = consensus_function
        else:
            self.consensus_function_name =  consensus_function.__name__

        self.floor_serialization = floor_serialization
        self.device_map = device_map
        self.eligible_devices = self._list_eligible_devices()
        self.connectivity_matrix = self._generate_base_connectivity_matrix()
        self.update_rate = update_rate

        self.reputations = None  ## initialise sensor reputations
        self.metadata = MetaData(
            device_id="Consensus Agent " + " ".join(self.floor_name),
            time_name='timestamp',
            time_unit='h',
            quantity_names=self.consensus_function_name + " " + " ".join(self.floor_name),
            quantity_units="\mu g/ m^3",
            misc="Consensus values " +" ".join(self.floor_name) + "in floor " + f"{floor_serialization['floor']:02d}",
        )
        super().init_parameters()

    def on_received_message(self, message):
        #number of input agents that are connected to the aggregator
        n_input_agents = len(self.get_attr("Inputs"))

        # store into buffer

        self.buffer.store(message["from"], {'data':message["data"], 'metadata':message["metadata"]})
        self.log_info(f"{self.buffer.values()}")

        ## compute mean of the latest value
        buffer_values = list(self.buffer.values())

        ## check 1: buffer len is the same
        if len(buffer_values) == n_input_agents:
            self.update_connectivity_matrix(buffer_values)
            self.update_reputations(buffer_values)
            reputation_timestamp = self.reputations['timestamp']
            for meastype, reputation_row in self.reputations['reputations'].items():
                packed_reputations = [self.send_output(data=[np.array([[reputation_timestamp, 0, val, 0]]), MetaData(
                    device_id=idx,
                    time_name='timestamp',
                    time_unit='h',
                    quantity_names=meastype,
                    quantity_units="None",
                )], channel="default") for idx, val in reputation_row.items()]
            self.buffer.clear()

    def _generate_base_connectivity_matrix(self):

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
            doornames = [qp[1].split(' ')[1]
                         for qp in quant_and_platform_names if 'door' in qp[0]]
            door_indices = [i for i, s in enumerate(quant_and_platform_names) if "door" in s[0]]
            door_states.update({dname:buffer_value['data'][ind][2].astype(int) for dname, ind in zip(doornames, door_indices)})

        # If the rooms are nested under a key (e.g., data['rooms']), adjust accordingly
        rooms = self.floor_serialization if isinstance(self.floor_serialization, list) else self.floor_serialization.get('zones', [])

        # 1. Get a list of all unique room IDs to define the matrix dimensions
        room_ids = sorted([room['guid'] for room in rooms])
        id_to_idx = {room_id: i for i, room_id in enumerate(room_ids)}

        room_to_sensors = {
            u_id: [dev for dev, rng in devices.items() if 'airvolume' in rng]
            for u_id, devices in self.device_map.items()
        }

        # Use a list to collect updates to avoid calling .loc in a loop
        # Format: (row_sensors, col_sensors, weight)
        updates = []

        for room in rooms:
            u_id = room.get('guid')
            u_sensors = room_to_sensors.get(u_id, [])

            if not u_sensors:
                continue

            for wall in room.get('walls', []):
                for neighbour in wall.get('neighbours', []):
                    v_id = neighbour.get('space_id')

                    # Skip if v_id is invalid or not in our target index
                    if v_id not in id_to_idx:
                        continue

                    v_sensors = room_to_sensors.get(v_id, [])
                    if not v_sensors:
                        continue

                    # 2. Simplify Weight Logic
                    doors = neighbour.get('door_ids_in_section', [])
                    if not doors:
                        weight = 0.25
                    else:
                        # Weight is 1.0 if ANY door is open (state != 2), else 0.5
                        weight = 1.0 if any(door_states.get(self._get_pname(d_id), 0) == 2 for d_id in doors) else 0.25

                    updates.append((u_sensors, v_sensors, weight))

        # 3. Batch update the DataFrame
        for u_sens, v_sens, weight in updates:
            self.connectivity_matrix.loc[u_sens, v_sens] = weight
            self.connectivity_matrix.loc[v_sens, u_sens] = weight

    def update_reputations(self, buffer_values):
        """
        Check if reputations have already been assigned. If not assign reputation "1" to all sensors.

        Else update reputations based on sensor readings and connectivity matrix. Only "participating"
        sensors are updated. i.e. Sensors not connected to any others, eg. via an open door or wall,
        retain their previous reputation values.

        :param buffer_values:
        """
        if self.reputations is None:
            reputation_data = defaultdict(dict)

            for buffer_value in buffer_values:
                metadata_list = buffer_value.get('metadata', [])
                if not metadata_list:
                    continue

                # Extract platform name from the first metadata element
                try:
                    misc_info = metadata_list[0]._metadata.get('misc', '')
                    platform_name = misc_info.split(' ')[1]
                except (IndexError, AttributeError):
                    continue


                if platform_name in self.eligible_devices:
                    for mdata in metadata_list:
                        if hasattr(mdata, '_metadata'):
                            device_id = mdata._metadata['device_id'].split('__')[1]
                            reputation_data[device_id][platform_name] = 1

            # 3. Create the DataFrame once at the end
            # Pandas handles a dict-of-dicts perfectly:
            # Outer keys = columns, Inner keys = index (rows)
            reputation_timestamp = self._reputation_timestamp(buffer_values)
            self.reputations = {'timestamp': reputation_timestamp, 'reputations': {device: pd.Series(reputations) for device, reputations in reputation_data.items()}}
        else:
            sensor_readings_dict = defaultdict(dict)
            sensor_uncertainties_dict = defaultdict(dict)
            #Retrieve sensor readings and timestamp from buffer values
            for buffer_value in buffer_values:
                for mdata, sdata in zip(buffer_value['metadata'], buffer_value['data']):
                    # Extract keys for readability
                    platform_key = mdata.metadata['misc'].split(' ')[-1]
                    reading_key = mdata.metadata['device_id'].split('__')[1]
                    measurement = float(sdata[2])
                    unc = float(sdata[3])

                    # This updates the inner dict instead of overwriting the outer one
                    if platform_key in self.eligible_devices:
                        sensor_readings_dict[platform_key][reading_key] = measurement
                        sensor_uncertainties_dict[platform_key][reading_key] = unc

            sensor_readings = pd.DataFrame(sensor_readings_dict)
            sensor_uncertainties = pd.DataFrame(sensor_uncertainties_dict)

            self.reputations['timestamp'] = self._reputation_timestamp(buffer_values)

            #update reputations based on sensor readings
            for sensor_reading_type, reputation_row in self.reputations['reputations'].items():
                sensor_readings_row = sensor_readings.loc[sensor_reading_type,reputation_row.index]
                reputation_weights = self.connectivity_matrix.loc[reputation_row.index, reputation_row.index].mul(reputation_row, axis=0)
                local_consensus = reputation_weights.mul(sensor_readings_row, axis=0).sum(axis=0) / reputation_weights.sum(axis=0)
                self.reputations['reputations'][sensor_reading_type] *= self._reputation_change_factor(sensor_readings_row, local_consensus)

    def _reputation_change_factor(self, sensor_readings, local_consensus, NaN_change_factor=0.9):
        sensor_deviations = np.abs(local_consensus - sensor_readings)
        reputation_change = sensor_deviations.combine(local_consensus,
                                                      lambda x, y: (np.exp(-25*x**2 / y**2)))
        rep_change_factor = self.update_rate + (1 - self.update_rate) * reputation_change
        return rep_change_factor.fillna(NaN_change_factor)

    def _list_eligible_devices(self):
        eligible_devices = []
        for room_id, devices in self.device_map.items():
            eligible_devices.extend([device for device, range in devices.items() if 'airvolume' in range])

        return eligible_devices

    def _reputation_timestamp(self, buffer_values):
        tstamps = [np.ravel(buffer_value['data'][:, 0]).tolist() for buffer_value in buffer_values]
        reputation_timestamp = set(list(itertools.chain.from_iterable(tstamps)))
        return reputation_timestamp

    def _get_pname(self, d_id):
        "Retrieve door sensor platform names from room id"
        room_devices = self.device_map.get(d_id[:7], {})
        target_val = [d_id.split('_')[2]]
        # Find the first key that matches the value
        return next((k for k, v in room_devices.items() if v == target_val), None)

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
    device_pathstring = 'Data/devices_by_room_and_floor.json'
    if Path(device_pathstring).is_file():
     json.dump(devices_by_room_and_floor, open(device_pathstring, 'w'))

    ser_pathstring = 'Data/building_ser.json'
    if Path(ser_pathstring).is_file():
        json.dump(building_ser, open(ser_pathstring, 'w'))

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
            setpoint_cols = [col for col in df.columns if 'setpoint' in col]
            valve_position_cols = [col for col in df.columns if 'valve_position' in col]
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
            df.drop(columns=offset_cols+setpoint_cols+valve_position_cols, inplace=True, errors='ignore')

            df_dict_internal[season_dict[basename[9:26]]] = df.set_index('timestamp')
        room_timeseries[key] = df_dict_internal
    #%%
    floor_agents = {}
    monitor_agents = {}
    platform_agent_dict = {}
    for floor in ['F01']:
        floor_agents[floor] = agent_network.add_agent(name="Consensus Agent " + floor, agentType=ConsensusAgent,
                                                      floor_name=floor,
                                                      floor_serialization=building_ser[floor], device_map=devices_by_room_and_floor[floor],
                                                      buffer_size=20)
        for zone in building_ser[floor]['zones']:
            rname = zone['guid']
            try:
                sdict = device_maps[rname]
                room_data = room_timeseries[rname]['summer']
                platform_agent_dict[rname] = []
                for device, location in sdict.items():
                    device_columns = [col for col in room_data.columns if device in col]
                    if not device_columns:
                        device_columns = [col for col in room_data.columns if any(loc in col for loc in location)]

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
                                                        agentType=MetrologicalMonitorAgent, buffer_size=50)

        floor_agents[floor].bind_output(monitor_agents[floor])

    agent_network.set_running_state()
    return agent_network

if __name__ == '__main__':
    building_network_demonstration()