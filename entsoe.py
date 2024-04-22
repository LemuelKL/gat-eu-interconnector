from dataclasses import dataclass
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from currency_converter import CurrencyConverter
from tqdm import tqdm
from config import (
    countries,
    dap_bidding_zones,
    interconnections,
    interconnections_edge_matrix,
)
from torch_geometric.data import Data
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch.utils.data import DataLoader
import torch


def load_data():

    # Check for cached data
    if os.path.exists("data"):
        print("Loading cached data")
        datetime_intersect = np.load("data/datetime_intersect.npy")
        node_features = np.load("data/node_features.npy")
        edge_indices = np.load("data/edge_indices.npy")
        edge_attributes = np.load("data/edge_attributes.npy")
        edge_labels = np.load("data/edge_labels.npy")

        node_features = torch.tensor(node_features, dtype=torch.float32)
        edge_indices = torch.tensor(edge_indices, dtype=torch.int64)

        edge_attributes = torch.tensor(edge_attributes, dtype=torch.float32)
        edge_labels = torch.tensor(edge_labels, dtype=torch.float32)

        data = [
            Data(
                x=node_features[i],
                edge_index=edge_indices[i],
                edge_attr=edge_attributes[i],
                y=edge_labels[i],
            )
            for i in range(len(datetime_intersect))
        ]
        return data

    # Load data from database
    load_dotenv()
    engine = create_engine(os.getenv("SQLALCHEMY_DATABASE_URI"))

    # Load flows from master table
    flow_df = pd.read_sql_table("flow_32", engine)
    flow_df = flow_df.set_index("DateTime")
    flow_df.fillna(0, inplace=True)

    # Load DAP from individual tables and convert GBP to EUR for UK
    c = CurrencyConverter(fallback_on_missing_rate=True)
    dap_df = pd.DataFrame()
    for country_id in countries.keys():
        df = pd.read_sql_table(f"{country_id}_dap", engine)
        if country_id == "UK":
            # Do currency conversion GBP -> EUR according to day
            df["DateTime"] = pd.to_datetime(df["DateTime"])
            df["EUR"] = df["DateTime"].apply(lambda x: c.convert(1, "GBP", "EUR", x))
            df.set_index("DateTime", inplace=True)
            df["0"] = df["0"] * df["EUR"]
            df.drop(columns=["EUR"], inplace=True)
            dap_df[country_id] = df
        else:
            dap_df[country_id] = df.set_index("DateTime")
    dap_df.ffill(inplace=True)
    dap_df.fillna(0, inplace=True)

    # Load grid load from individual tables
    load_df = pd.DataFrame()
    for country_id in countries.keys():
        load_df[country_id] = pd.read_sql_table(f"{country_id}_load", engine).set_index(
            "DateTime"
        )
    load_df.ffill(inplace=True)
    # Fille NaN with mean of the column
    load_df.fillna(load_df.mean(), inplace=True)

    biomass_df = pd.DataFrame()
    fossil_brown_coal_df = pd.DataFrame()
    fossil_coal_derived_gas_df = pd.DataFrame()
    fossil_gas_df = pd.DataFrame()
    fossil_hard_coal_df = pd.DataFrame()
    fossil_oil_df = pd.DataFrame()
    hydro_pumped_storage_df = pd.DataFrame()
    hydro_run_of_river_and_poundage_df = pd.DataFrame()
    hydro_water_reservoir_df = pd.DataFrame()
    nuclear_df = pd.DataFrame()
    other_df = pd.DataFrame()
    other_renewable_df = pd.DataFrame()
    solar_df = pd.DataFrame()
    waste_df = pd.DataFrame()
    wind_offshore_df = pd.DataFrame()
    wind_onshore_df = pd.DataFrame()
    geothermal_df = pd.DataFrame()
    fossil_peat_df = pd.DataFrame()

    gen_types = [
        "Biomass",
        "Fossil Brown coal/Lignite",
        "Fossil Coal-derived gas",
        "Fossil Gas",
        "Fossil Hard coal",
        "Fossil Oil",
        "Hydro Pumped Storage",
        "Hydro Run-of-river and poundage",
        "Hydro Water Reservoir",
        "Nuclear",
        "Other",
        "Other renewable",
        "Solar",
        "Waste",
        "Wind Offshore",
        "Wind Onshore",
        "Geothermal",
        "Fossil Peat",
    ]

    for country_id in countries.keys():
        this_cty_gen_df = pd.read_sql_table(f"{country_id}_gen", engine).set_index(
            "DateTime"
        )
        biomass_df[country_id] = this_cty_gen_df["Biomass"]
        fossil_brown_coal_df[country_id] = this_cty_gen_df["Fossil Brown coal/Lignite"]
        fossil_coal_derived_gas_df[country_id] = this_cty_gen_df[
            "Fossil Coal-derived gas"
        ]
        fossil_gas_df[country_id] = this_cty_gen_df["Fossil Gas"]
        fossil_hard_coal_df[country_id] = this_cty_gen_df["Fossil Hard coal"]
        fossil_oil_df[country_id] = this_cty_gen_df["Fossil Oil"]
        hydro_pumped_storage_df[country_id] = this_cty_gen_df["Hydro Pumped Storage"]
        hydro_run_of_river_and_poundage_df[country_id] = this_cty_gen_df[
            "Hydro Run-of-river and poundage"
        ]
        hydro_water_reservoir_df[country_id] = this_cty_gen_df["Hydro Water Reservoir"]
        nuclear_df[country_id] = this_cty_gen_df["Nuclear"]
        other_df[country_id] = this_cty_gen_df["Other"]
        other_renewable_df[country_id] = this_cty_gen_df["Other renewable"]
        solar_df[country_id] = this_cty_gen_df["Solar"]
        waste_df[country_id] = this_cty_gen_df["Waste"]
        wind_offshore_df[country_id] = this_cty_gen_df["Wind Offshore"]
        wind_onshore_df[country_id] = this_cty_gen_df["Wind Onshore"]
        geothermal_df[country_id] = this_cty_gen_df["Geothermal"]
        fossil_peat_df[country_id] = this_cty_gen_df["Fossil Peat"]

    biomass_df.fillna(0, inplace=True)
    fossil_brown_coal_df.fillna(0, inplace=True)
    fossil_coal_derived_gas_df.fillna(0, inplace=True)
    fossil_gas_df.fillna(0, inplace=True)
    fossil_hard_coal_df.fillna(0, inplace=True)
    fossil_oil_df.fillna(0, inplace=True)
    hydro_pumped_storage_df.fillna(0, inplace=True)
    hydro_run_of_river_and_poundage_df.fillna(0, inplace=True)
    hydro_water_reservoir_df.fillna(0, inplace=True)
    nuclear_df.fillna(0, inplace=True)
    other_df.fillna(0, inplace=True)
    other_renewable_df.fillna(0, inplace=True)
    solar_df.fillna(0, inplace=True)
    waste_df.fillna(0, inplace=True)
    wind_offshore_df.fillna(0, inplace=True)
    wind_onshore_df.fillna(0, inplace=True)
    geothermal_df.fillna(0, inplace=True)
    fossil_peat_df.fillna(0, inplace=True)

    datetime_intersect = (
        flow_df.index.intersection(dap_df.index)
        .intersection(load_df.index)
        .intersection(biomass_df.index)
        .intersection(fossil_brown_coal_df.index)
        .intersection(fossil_coal_derived_gas_df.index)
        .intersection(fossil_gas_df.index)
        .intersection(fossil_hard_coal_df.index)
        .intersection(fossil_oil_df.index)
        .intersection(hydro_pumped_storage_df.index)
        .intersection(hydro_run_of_river_and_poundage_df.index)
        .intersection(hydro_water_reservoir_df.index)
        .intersection(nuclear_df.index)
        .intersection(other_df.index)
        .intersection(other_renewable_df.index)
        .intersection(solar_df.index)
        .intersection(waste_df.index)
        .intersection(wind_offshore_df.index)
        .intersection(wind_onshore_df.index)
        .intersection(geothermal_df.index)
        .intersection(fossil_peat_df.index)
    )
    # Check if datetime_intersect is monotonically increasing
    assert all(
        datetime_intersect[i] < datetime_intersect[i + 1]
        for i in range(len(datetime_intersect) - 1)
    )

    # Create temporal features based on datetime_intersect
    temporal_hour_df = pd.DataFrame(index=datetime_intersect)
    temporal_dow_df = pd.DataFrame(index=datetime_intersect)
    temporal_month_df = pd.DataFrame(index=datetime_intersect)
    temporal_doy_df = pd.DataFrame(index=datetime_intersect)
    for country_id in countries.keys():
        temporal_hour_df[country_id] = datetime_intersect.hour
        temporal_dow_df[country_id] = datetime_intersect.dayofweek
        temporal_month_df[country_id] = datetime_intersect.month
        temporal_doy_df[country_id] = datetime_intersect.dayofyear

    flow_df = flow_df.loc[datetime_intersect]
    dap_df = dap_df.loc[datetime_intersect]
    load_df = load_df.loc[datetime_intersect]
    biomass_df = biomass_df.loc[datetime_intersect]
    fossil_brown_coal_df = fossil_brown_coal_df.loc[datetime_intersect]
    fossil_coal_derived_gas_df = fossil_coal_derived_gas_df.loc[datetime_intersect]
    fossil_gas_df = fossil_gas_df.loc[datetime_intersect]
    fossil_hard_coal_df = fossil_hard_coal_df.loc[datetime_intersect]
    fossil_oil_df = fossil_oil_df.loc[datetime_intersect]
    hydro_pumped_storage_df = hydro_pumped_storage_df.loc[datetime_intersect]
    hydro_run_of_river_and_poundage_df = hydro_run_of_river_and_poundage_df.loc[
        datetime_intersect
    ]
    hydro_water_reservoir_df = hydro_water_reservoir_df.loc[datetime_intersect]
    nuclear_df = nuclear_df.loc[datetime_intersect]
    other_df = other_df.loc[datetime_intersect]
    other_renewable_df = other_renewable_df.loc[datetime_intersect]
    solar_df = solar_df.loc[datetime_intersect]
    waste_df = waste_df.loc[datetime_intersect]
    wind_offshore_df = wind_offshore_df.loc[datetime_intersect]
    wind_onshore_df = wind_onshore_df.loc[datetime_intersect]
    geothermal_df = geothermal_df.loc[datetime_intersect]
    fossil_peat_df = fossil_peat_df.loc[datetime_intersect]

    edges = np.array(interconnections_edge_matrix)
    # Map edge names to indices
    edge_names = np.unique(edges)
    edge_map = {edge: i for i, edge in enumerate(edge_names)}
    edge_indices = np.array([edge_map[edge] for edge in edges.flatten()]).reshape(
        edges.shape
    )
    # Repeat edge indices for each datetime
    edge_indices = np.repeat(
        edge_indices[np.newaxis, :, :],
        len(datetime_intersect),
        axis=0,
    )
    n_edges = edges.shape[1]

    # Edge labels (flow) of shape (n_datetime, n_edges, 1)
    edge_labels = np.array(flow_df)
    # print(edge_labels.shape)
    edge_labels = np.reshape(
        edge_labels, (len(datetime_intersect), edge_labels.shape[1], 1)
    )
    edge_attributes = np.copy(edge_labels)

    # Node features (dap, load) of shape (n_datetime, n_nodes, n_node_features)
    node_features = np.stack(
        [
            dap_df.to_numpy(),
            load_df.to_numpy(),
            biomass_df.to_numpy(),
            fossil_brown_coal_df.to_numpy(),
            fossil_coal_derived_gas_df.to_numpy(),
            fossil_gas_df.to_numpy(),
            fossil_hard_coal_df.to_numpy(),
            fossil_oil_df.to_numpy(),
            hydro_pumped_storage_df.to_numpy(),
            hydro_run_of_river_and_poundage_df.to_numpy(),
            hydro_water_reservoir_df.to_numpy(),
            nuclear_df.to_numpy(),
            other_df.to_numpy(),
            other_renewable_df.to_numpy(),
            solar_df.to_numpy(),
            waste_df.to_numpy(),
            wind_offshore_df.to_numpy(),
            wind_onshore_df.to_numpy(),
            geothermal_df.to_numpy(),
            fossil_peat_df.to_numpy(),
            temporal_hour_df.to_numpy(),
            temporal_dow_df.to_numpy(),
            temporal_month_df.to_numpy(),
            temporal_doy_df.to_numpy(),
        ],
        axis=-1,
    )
    n_nodes = node_features.shape[1]

    assert (
        len(datetime_intersect)
        == edge_indices.shape[0]
        == edge_labels.shape[0]
        == edge_attributes.shape[0]
        == node_features.shape[0]
    )

    node_features = torch.tensor(node_features, dtype=torch.float32)
    edge_indices = torch.tensor(edge_indices, dtype=torch.int64)
    edge_attributes = torch.tensor(edge_attributes, dtype=torch.float32)
    edge_labels = torch.tensor(edge_labels, dtype=torch.float32)

    # Normalize features (time, node/edge, features)
    node_features = (
        node_features - node_features.mean(axis=(0, 1))
    ) / node_features.std(axis=(0, 1))
    edge_attributes = (
        edge_attributes - edge_attributes.mean(axis=(0, 1))
    ) / edge_attributes.std(axis=(0, 1))
    edge_labels = (edge_labels - edge_labels.mean(axis=(0, 1))) / edge_labels.std(
        axis=(0, 1)
    )

    print(
        node_features.shape,
        edge_indices.shape,
        edge_attributes.shape,
        edge_labels.shape,
    )

    # Save data
    os.makedirs("data", exist_ok=True)
    np.save("data/datetime_intersect.npy", datetime_intersect)
    np.save("data/node_features.npy", node_features)
    np.save("data/edge_indices.npy", edge_indices)
    np.save("data/edge_attributes.npy", edge_attributes)
    np.save("data/edge_labels.npy", edge_labels)

    data = [
        Data(
            x=node_features[i],
            edge_index=edge_indices[i],
            edge_attr=edge_attributes[i],
            y=edge_labels[i],
        )
        for i in range(len(datetime_intersect))
    ]

    return data


def get_dap_dataloader(window_size, future_steps, batch_size=50):
    data = load_data()

    len_data = len(data)
    all_x = []
    all_edge_weights = []
    all_edge_index = []
    all_y = []
    for i in range(0, len_data - window_size - future_steps):
        window_data = data[i : i + window_size]
        future_data = data[
            i + window_size + future_steps - 1
        ]  # one single observation at future_steps steps ahead

        x = torch.stack([d.x for d in window_data])
        edge_weights = torch.stack([d.edge_attr for d in window_data])
        # take the average of edge weights and retain its shape
        avg_edge_weights = torch.mean(edge_weights, dim=0, keepdim=True)
        edge_weights = avg_edge_weights.repeat(window_size, 1, 1)
        edge_index = torch.stack([d.edge_index for d in window_data])
        y = future_data.x[:, 0]  # flow

        # print(x.shape, edge_weights.shape, edge_index.shape, y.shape)
        all_x.append(x)
        all_edge_weights.append(edge_weights)
        all_edge_index.append(edge_index)
        all_y.append(y)

    all_x = torch.stack(all_x)
    all_edge_weights = torch.stack(all_edge_weights)
    all_edge_index = torch.stack(all_edge_index)
    all_y = torch.stack(all_y)

    print(all_x.shape, all_edge_weights.shape, all_edge_index.shape, all_y.shape)

    dataset = torch.utils.data.TensorDataset(
        all_x, all_edge_weights, all_edge_index, all_y
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


if __name__ == "__main__":
    # dataloader = get_dap_dataloader(24, 4)
    # print(dataloader)
    # for x, edge_weights, edge_index, y in dataloader:
    #     print(x.shape, edge_weights.shape, edge_index.shape, y.shape)
    #     break
    data = load_data()
    print(len(data))
