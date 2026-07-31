from typing import Union, Tuple, Optional, Any, List

import numpy as np
import pandas as pd

from AirQual_Umbrella.load_data_multiindex import load_data
from agentMET4FOF.agents import MetrologicalAgent, MetrologicalMonitorAgent
from agentMET4FOF.metrological_streams import MetrologicalDataStreamMET4FOF
from agentMET4FOF.network import AgentNetwork
from agentMET4FOF.utils import Backend

df_all, board_coords = load_data(return_coords=True)

class Sensor(MetrologicalDataStreamMET4FOF):
    """Streaming data from a sensor located on a specified platform

    Parameters
    ----------
    platform_name : str, optional
        id of the platform on which the sensing unit is located
    uncertainty : float
        measurement uncertainty of the sensor. usually found on calibration certificate or tech specs
    output_unit : str
        SI unit of the sensor output
    sensor_type : str, optional
        type of sensor based on what is being measured
    data_stream : Union[List, DataFrame, np.ndarray]
        data stream of sensor measurements indexed by time, e.g. timestamps
    """


    def __init__(
            self, uncertainty: float = 0, name="default_name", platform_name=None, sensor_type=None,
                        output_quantities: List[str] = None, output_unit=None,
                        data_stream: Union[List, pd.DataFrame, np.ndarray]=None, **kwargs,
    ):
        super().__init__(
            value_unc=uncertainty, time_unc=0
        )
        self.set_metadata(
            device_id=name,
            time_name="timestamp",
            time_unit="s",
            quantity_names='Relative Humidity',
            quantity_units='Percent',
            misc="Testing sensor as datastream",
        )
        self.value_unc = uncertainty
        self.time_unc = 0

        self.value_unc = uncertainty
        self.time_unc = 0
        self.output_quantities = output_quantities
        self.output_unit = output_unit
        self.platform = platform_name
        self.sensor_type = sensor_type
        self.sensor_name = name

        if data_stream is not None:
            tstamps = data_stream.index.astype(str).to_numpy().reshape(-1, 1)
            quants = data_stream.to_numpy()
            self.set_data_source(quantities=quants, time=tstamps)
        else:
            # tstamps = df_all.loc[('J', '01')][['T']].index.astype(int).to_numpy() / 10 ** 9
            tstamps = df_all.loc[('J', '01')][['T']].index.astype(str).to_numpy()
            self.set_data_source(quantities=df_all.loc[('J', '01')][['T']].to_numpy(),
                                 time=tstamps.reshape(-1, 1))

class SensorPlatformOld(MetrologicalAgent):
    """A metrological agent representing a platform hosting one or more sensors in an IoT network
     """

    def init_parameters(self, streams: Union[Sensor(), List[Sensor()]] = None, platform_data: pd.DataFrame = None, position: Union[Tuple[float, float], str] = None, id: str = None, **kwargs):
        """Initialize the sensor agent

         Parameters
         -----------
            :param position: The location of the sensor  given either by explicit geographical coordinates or a string descriptor
            :type position:  Union[Tuple[float, float], str]
            :param id: The id of the sensor platform
            :type id: str
         """

        self.position = position
        self.name = id
        super().init_parameters()
        if streams is not None:
            self._streams = streams
        elif platform_data is not None:
            self._streams = []
            for col in platform_data.columns:
                self._streams.append(Sensor(platform_name=id, name=col, data_stream=platform_data[col]))

        for stream in self._streams:
            self.set_output_data(channel=stream.name, metadata=stream.metadata)

    def agent_loop(self):
        """Model the agent's behaviour

        On state *Running* the agent will extract sample by sample the input data
        streams content and push it via invoking :py:method:`AgentMET4FOF.send_output`.
        """
        if self.current_state == "Running":
            for stream in self.streams:
                self.set_output_data(channel=stream.name, metadata=self.stream.next_sample())
            super().agent_loop()

class SensorPlatform(MetrologicalAgent):
    """An agent streaming a sine signal

    Takes samples from an instance of :py:class:`MetrologicalSineGenerator` and pushes
    them sample by sample to connected agents via its output channel.
    """

    # The datatype of the stream will be MetrologicalSineGenerator.
    _stream: MetrologicalDataStreamMET4FOF

    def init_parameters(
            self,
            streams: Union[MetrologicalDataStreamMET4FOF, List[MetrologicalDataStreamMET4FOF]] = Sensor(),
            platform_data: pd.DataFrame = None,
            position: Union[Tuple[float, float], str] = None,
            platform_id: str = None,
            **kwargs
    ):
        """Initialize the sensor platform agent

         Parameters
         -----------
            :param position: The location of the sensor  given either by explicit geographical coordinates or a string descriptor
            :type position:  Union[Tuple[float, float], str]
            :param id: The id of the sensor platform
            :type id: str
         """
        """Initialize the input data stream

        Parameters
        ----------
        signal : MetrologicalDataStreamMET4FOF
            the underlying signal for the generator
        """
        self._stream = streams
        super().init_parameters()
        self.set_output_data(channel="default", metadata=self._stream.metadata)

    def agent_loop(self):
        """Model the agent's behaviour

        On state *Running* the agent will extract sample by sample the input
        datastream's content and push it into its output buffer.
        """
        if self.current_state == "Running":
            self.set_output_data(channel="default", data=self._stream.next_sample())
            super().agent_loop()


def demonstrate_aq_agent():
    # Start agent network server.
    agent_network = AgentNetwork(backend=Backend.MESA)

    # platform_ags = {}
    # for board_key, coords in board_coords.items():
    #     platform_ags[board_key]  = agent_network.add_agent(name=board_key[0]+board_key[1], agentType=SensorPlatform)
    # platform_ags[board_key].init_parameters(position=coords, id=board_key[0] + board_key[1],
    #                                         platform_data=df_all.loc[board_key])

    signal = Sensor(uncertainty=0.1, data_stream=df_all.loc[('P', '02')][['CO']])
    source_name = signal.metadata.metadata["device_id"]
    source_agent = agent_network.add_agent(name=source_name, agentType=SensorPlatform)
    source_agent.init_parameters(streams=signal)
    monitor_agent = agent_network.add_agent(name="Monitor Agent", agentType=MetrologicalMonitorAgent, buffer_size=50)
    # agent_network.bind_agents(platform_ags[('V', '01')], monitor_agent)
    # setup health coalition group
    # agent_network.add_coalition(
    #     "REDUNDANT_SENSORS", [gen_agent_1, gen_agent_2, monitor_agent]
    # )

    # Set all agents' states to "Running".
    source_agent.bind_output(monitor_agent)
    agent_network.set_running_state()

    # Allow for shutting down the network after execution
    return agent_network


if __name__ == '__main__':
    demonstrate_aq_agent()