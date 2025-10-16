import numpy as np
import torch
from collections import OrderedDict
from scipy.spatial import distance
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from geopy.distance import geodesic
from metpy.units import units
import metpy.calc as mpcalc
from haversine import haversine, Unit
import pandas as pd
from utils.geo_utils import get_elevation, interpolate_points
import os


get_elevation = np.vectorize(get_elevation)

class Graph:
    def __init__(self, sensor_path, srtm_path, dist_thres=300, alti_thres=1200, middle=15):
        """
        dist_thres: km
        alti_thres: m
        adj_mx: N x N
        edge_index: M x 2
        edge_attr: M x D
        node_attr: N x D
        """
        self.dist_thres = dist_thres
        self.alti_thres = alti_thres
        self.middle = middle

        self.sensor = pd.read_csv(sensor_path)
        self.nodes_num = self.sensor.shape[0]
        self.srtm_path = srtm_path

        self.query_altitude()
        self.adj_mx = self._gen_edges()
        self.edge_index, self.edge_attr, self.node_attr = self._gen_attr()

    def query_altitude(self):
        lat = self.sensor['latitude'].values
        lon = self.sensor['longitude'].values
        alt = get_elevation(lat, lon, self.srtm_path)
        self.sensor['altitude'] = alt

    def _gen_edges(self):
        points = self.sensor[['latitude', 'longitude']].values  # N x 2

        dist_km = np.zeros((self.nodes_num, self.nodes_num))
        for i, A in enumerate(points):
            for j, B in enumerate(points):
                dist_km[i][j] = haversine(A, B, unit=Unit.KILOMETERS)
        dist_adj = np.zeros((self.nodes_num, self.nodes_num), dtype=np.uint8)
        dist_adj[dist_km <= self.dist_thres] = 1
        np.fill_diagonal(dist_adj, 0)

        alt_adj = np.zeros_like(dist_adj)
        edge_index, _ = dense_to_sparse(torch.tensor(dist_adj))
        edges = edge_index.t().tolist()
        edges = [sorted(e) for e in edges]
        unique_edges = list(map(list, set(map(tuple, edges))))
        unique_edge_index = np.array(unique_edges).T

        M = unique_edge_index.shape[1]
        middle_points = np.zeros((M, self.middle, 2))
        for i in range(M):
            src = self.sensor[['latitude', 'longitude']].iloc[unique_edge_index[0][i]]
            dest = self.sensor[['latitude', 'longitude']].iloc[unique_edge_index[1][i]]
            middle = interpolate_points(src, dest, num_points=self.middle)
            middle_points[i] = middle
        middle_points_alt = get_elevation(middle_points[:, :, 0], middle_points[:, :, 1], self.srtm_path)
        max_alt = np.max(middle_points_alt[:, [0, -1]], axis=1)[:, None]   # M
        alt_gap = np.max(middle_points_alt - max_alt, axis=1)
        for i in range(M):
            src_index = unique_edge_index[0][i]
            dest_index = unique_edge_index[1][i]
            if alt_gap[i] <= self.alti_thres:
                alt_adj[src_index][dest_index] = 1
                alt_adj[dest_index][src_index] = 1
        adj = dist_adj * alt_adj

        return adj

    def _gen_attr(self):
        edge_index, _ = dense_to_sparse(torch.tensor(self.adj_mx))
        edge_index = edge_index.numpy()
        node_attr = self.sensor['altitude'].values
        edge_attr = []
        M = edge_index.shape[1]

        for i in range(M):
            src_index = edge_index[0][i]
            dest_index = edge_index[1][i]
            src = self.sensor[['latitude', 'longitude']].iloc[src_index]
            dest = self.sensor[['latitude', 'longitude']].iloc[dest_index]
            dist_km = haversine(src, dest, unit=Unit.KILOMETERS)
            # diff_dist = np.exp(-dist_km) / 0.001
            diff_dist = 1 / dist_km

            v, u = src['latitude'] - dest['latitude'], src['longitude'] - dest['longitude']
            u = u * units.meter / units.second
            v = v * units.meter / units.second
            direction = mpcalc.wind_direction(u, v)._magnitude

            edge_attr.append([diff_dist, dist_km, direction])

        edge_attr = np.array(edge_attr)
        edge_index = edge_index.T
        node_attr = node_attr[:, None]

        return edge_index, edge_attr, node_attr

    def save_npz(self, save_path):
        save_path = os.path.join(save_path, 'graph_data.npz')
        np.savez(save_path, adj_mx=self.adj_mx, node_attr=self.node_attr,
                 edge_index=self.edge_index, edge_attr=self.edge_attr)

# KnowAir Example
if __name__ == "__main__":
    dataset_name = "KnowAir"
    graph = Graph("../dataset/{}/station.csv".format(dataset_name), "../dataset/srtm",
                  dist_thres=300, alti_thres=1200, middle=15)
    graph.save_npz('../dataset/{}'.format(dataset_name))
    print(graph.adj_mx.shape)
    print(graph.edge_index.shape)
    print(graph.edge_attr.shape)
    print(graph.node_attr.shape)