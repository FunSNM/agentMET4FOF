from typing import Tuple, Union, List

import numpy as np
import pandas as pd

from agentMET4FOF.agents.metrological_base_agents import MetrologicalAgent, MetrologicalMonitorAgent
from agentMET4FOF.agents.metrological_signal_agents import MetrologicalGeneratorAgent
from agentMET4FOF.network import AgentNetwork
from agentMET4FOF.streams.metrological_base_streams import MetrologicalDataStreamMET4FOF


df_heatmeter = pd.read_csv("C://Users//singh04//Desktop//measurementsTimetable.txt")
df_heatmeter_dict = {}
unique_meter_array = df_heatmeter['Meter'].unique()
for k in unique_meter_array:
    df_specific = df_heatmeter[df_heatmeter['Meter']==k].drop('Meter', axis=1)
    df_heatmeter_dict.update({k:df_specific.set_index('Time')})

df_heatmeter_multiindex = pd.concat(df_heatmeter_dict)
df_heatmeter_multiindex.index = pd.MultiIndex.from_tuples(df_heatmeter_multiindex.index)
df_heatmeter_multiindex.index.names = ['Meter', 'Timestamp']


class SensorOnPlatform(MetrologicalDataStreamMET4FOF):
    def __init__(
        self, uncertainty: float =0, platform_name=None, sensor_type=None,
        output_unit=None, data_stream: Union[List, pd.DataFrame, np.ndarray]=None
    ):
        self.uncertainty = uncertainty
        self.output_unit = output_unit
        self.platform = platform_name
        self.sensor_type = sensor_type

        # store data_stream directly for later use
        self.data_stream = data_stream  

        super(SensorOnPlatform, self).__init__(value_unc=self.uncertainty, time_unc=0)
        self.set_metadata(
            self.platform+'_'+data_stream.columns.values[0],
            "time",
            "h",
            self.sensor_type,
            output_unit,
            "Data Stream from Heat Meter Readings",
        )
        self.set_data_source(quantities=data_stream, time=pd.DataFrame(data_stream.index.values))



# NEW CLASS: MultiSensorOnPlatform for hanling multiple sensors
# This is the key change: combines multiple SensorOnPlatform objects
# into a single signal stream for one agent
# -------------------------------------------------------------------

class MultiSensorOnPlatform(MetrologicalDataStreamMET4FOF):
    """Handles multiple sensors on the same platform as a single stream."""
    def __init__(self, sensors: List[SensorOnPlatform]):
        self.sensors = sensors
        self.uncertainty = np.mean([s.uncertainty for s in sensors])
        super().__init__(value_unc=self.uncertainty, time_unc=0)

        # Combine metadata
        self.set_metadata(
            "HeatMeterAgent", "time", "h", "Temperature", "°C",
            "Combined data from multiple sensors"
        )

        # Combine all quantities into a single DataFrame
        quantities_list = [s.data_stream for s in sensors]  # use stored data_stream
        quantities = pd.concat(quantities_list, axis=1)
        quantities.columns = [s.metadata.metadata['device_id'] for s in sensors]

        # Use time index from first sensor
        times = pd.DataFrame(sensors[0].data_stream.index)
        self.set_data_source(quantities=quantities, time=times)
        



def demonstrate_metrological_stream():
    """Demonstrate an agent network with two metrologically enabled agents

    The agents are defined as objects of the :class:`MetrologicalGeneratorAgent`
    class whose outputs are bound to a single monitor agent.

    The metrological agents generate signals from a sine wave and a multiwave generator
    source.

    Returns
    -------
    :class:`AgentNetwork`
        The initialized and running agent network object
    """
    # start agent network server
    agent_network = AgentNetwork(dashboard_modules=True, ip_addr='127.0.0.1')

    # Initialize two SensorOnPlatform objects (low and high temperature)
    signal_tempLow = SensorOnPlatform(
        uncertainty=1.5, platform_name='HeatMeter', sensor_type='Temperature',
        output_unit='°C', data_stream=df_heatmeter_multiindex.loc[11][['tempLow']]
    )
    signal_tempHigh = SensorOnPlatform(
        uncertainty=1.0, platform_name='HeatMeter', sensor_type='Temperature',
        output_unit='°C', data_stream=df_heatmeter_multiindex.loc[11][['tempHigh']]
    )

    # -------------------------------------------------------------------
    # Key Change: Combine the two sensors into a single MultiSensorOnPlatform
    # -------------------------------------------------------------------
    combined_sensor = MultiSensorOnPlatform([signal_tempLow, signal_tempHigh])


  # Create a single MetrologicalGeneratorAgent to handle both sensors
    combined_agent = agent_network.add_agent(
        name="HeatMeterAgent",
        agentType=MetrologicalGeneratorAgent
    )
    combined_agent.init_parameters(signal=combined_sensor)  # pass single combined signal


    # Initialize metrologically enabled plotting agent.
    monitor_agent = agent_network.add_agent(
        "MonitorAgent",
        agentType=MetrologicalMonitorAgent,
        buffer_size=50,
    )

     # Bind the combined agent to the monitor
    combined_agent.bind_output(monitor_agent)

    # Set all agents to running
    agent_network.set_running_state()

    return agent_network



if __name__ == "__main__":
    demonstrate_metrological_stream()
