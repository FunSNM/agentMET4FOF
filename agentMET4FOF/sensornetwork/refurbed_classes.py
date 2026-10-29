import datetime
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
from time_series_metadata.scheme import MetaData

from agentMET4FOF.agents import MetrologicalAgent
from agentMET4FOF.metrological_streams import MetrologicalDataStreamMET4FOF


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
            self, uncertainty: float = 0, name="default_name", platform_name="default_platform", sensor_type="default_type",
                        output_quantities: List[str] = None, output_unit=None,
                        data_stream: Union[List, pd.DataFrame, pd.Series, np.ndarray]=None,
                        quantity_units=None, **kwargs,
    ):
        super().__init__(
            value_unc=uncertainty, time_unc=0
        )
        if type(data_stream) is pd.DataFrame:
            self.set_metadata(
                device_id=name,
                time_name=data_stream.index.name,
                time_unit="h",
                quantity_names=data_stream.columns,
                quantity_units=quantity_units,
                misc='Platform ' + platform_name,
            )
        else:
            self.set_metadata(
                device_id=name,
                time_name="time",
                time_unit="s",
                quantity_names="",
                quantity_units="",
                misc='',
            )

        self.value_unc = uncertainty
        self.time_unc = 0
        self.output_quantities = output_quantities
        self.output_unit = output_unit
        self.platform = platform_name
        self.sensor_type = sensor_type
        self.sensor_name = name

        if data_stream is not None:
            timestamps = data_stream.index.astype(str).to_numpy().reshape(-1,1)
            quants = data_stream.to_numpy().reshape(-1,1)
            self.set_data_source(quantities=quants, time=timestamps)
        else:
            base = pd.Timestamp.today()
            n_days = 7
            timestamps = np.array([base + datetime.timedelta(days=x) for x in range(n_days)]).reshape(-1,1)
            quants = np.ones_like(timestamps)
            self.set_data_source(quantities=quants, time=timestamps)

    @property
    def name(self):
        return self.sensor_name



class SensorPlatform(MetrologicalAgent):
    """An agent streaming a sine signal

    Takes samples from an instance of :py:class:`MetrologicalSineGenerator` and pushes
    them sample by sample to connected agents via its output channel.
    """

    # The datatype of the stream will be MetrologicalSineGenerator.
    _streams: MetrologicalDataStreamMET4FOF

    def init_parameters(
            self,
            streams: Union[MetrologicalDataStreamMET4FOF, List[MetrologicalDataStreamMET4FOF]] = None,
            platform_timeseries: pd.DataFrame = None,
            sensor_uncs: Union[np.array, List[float], dict] = None,
            position: Union[Tuple[float, float], str] = None,
            platform_id: str = None,
            platform_type: str = None,
            **kwargs
    ):
        """Initialize the sensor platform agent either directly by specifying the sensor `streams` or the `platform_timeseries`
        which will be used to initialize the sensor streams

         Parameters
         -----------

            :param streams: The sensor streams available on the platform
            :type streams:  Union[MetrologicalDataStreamMET4FOF, List[MetrologicalDataStreamMET4FOF]]
            :param platform_timeseries: Historical time series data from the platform being modeled with labeled sensor measurements
            :type platform_timeseries: pd.DataFrame
            :type sensor_uncs: Union[np.array, List[float], dict]
            :param sensor_uncs: The uncertainties of the sensors on the platform
             :param position: The location of the sensor  given either by explicit geographical coordinates or a string descriptor
            :type position:  Union[Tuple[float, float], str]
            :param platform_id: The id of the sensor platform
            :type platform_id: str
        """
        super().init_parameters()

        self.platform_id = platform_id
        self.position = position
        if streams is not None:
            self._streams = streams
            if type(self._streams) is Sensor:
                self._streams = [self._streams]
        elif platform_timeseries is not None:
            self._streams = []
            if sensor_uncs is None:
                sensor_uncs = {col: 0 for col in platform_timeseries.columns}
            for col in platform_timeseries.columns:
                self._streams.append(Sensor(uncertainty=sensor_uncs[col], name=col, platform_name=self.platform_id, data_stream=platform_timeseries[[col]]))


    def agent_loop(self):
        """Model the agent's behaviour

        On state *Running* the agent will extract sample by sample the input
        datastream's content and push it into its output buffer.
        """
        if self.current_state == "Running":
            for stream in self._streams:
                self.set_output_data(channel="default", data=stream.next_sample(), metadata=stream.metadata)
            super().agent_loop()

    @property
    def location(self):
        return self.position

class AggregatorAgent(MetrologicalAgent):
    ## Compute average of the values received

    def init_parameters(self, aggregate_keys=["O3"], data_aggregation_function=np.nanmean, aggregator_description=None):
        self.aggregate_keys = aggregate_keys
        self.aggregation_function = data_aggregation_function
        if type(data_aggregation_function) == str:
            self.agg_function_name = data_aggregation_function
        else:
            self.agg_function_name =  data_aggregation_function.__name__

        self.aggregated_values = []
        self.metadata = MetaData(
            device_id="Aggregator Agent " + " ".join(self.aggregate_keys),
            time_name='timestamp',
            time_unit='h',
            quantity_names=self.agg_function_name + " " + " ".join(self.aggregate_keys),
            quantity_units="\mu g/ m^3",
            misc="Aggregation of "+" ".join(self.aggregate_keys)+ "in Cluster "+aggregator_description,
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


            aggregated_values, aggregated_uncertainties = self.agg_function(buffer_values)
            quantity_names = [mdata._metadata['quantity_names'][0] for mdata in buffer_values[0]['metadata']]
            for key in self.aggregate_keys:
                agg_index = quantity_names.index(key)
                self.send_output(data=[np.array([[buffer_values[0]['data'][0,0], 0, aggregated_values[agg_index],
                                                  aggregated_uncertainties[agg_index]]]), self.metadata], channel="default")

            self.buffer.clear()

    def agg_function(self, buffer_values):
        buffer_array = np.concatenate([[np.array(buffer_values[0]['data'])], [np.array(buffer_values[1]['data'])],
                                       [np.array(buffer_values[2]['data'])]])
        aggregated_values = np.nanmean(buffer_array[:, :, 2].astype(float), axis=0)
        n_notnan_values = np.sum(~np.isnan(buffer_array[:, :, 2].astype(float)), axis=0)
        aggregated_uncertainties = np.divide(np.nanstd(buffer_array[:, :, 3].astype(float), axis=0, mean=0),
                                             np.sqrt(n_notnan_values), where=n_notnan_values > 0)

        return aggregated_values, aggregated_uncertainties