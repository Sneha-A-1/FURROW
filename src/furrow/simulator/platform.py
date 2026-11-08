"""Layer 6: heterogeneous execution platform and data-access cost model.

All infrastructure parameters below are documented modelling assumptions (they are
not part of the agricultural data set); experiments vary the important ones.

Execution-time model for task tau with retrieval variant v on node n:

    e(tau, n) = ceil(parts_v / c_io) * l_req(n, tier_v) + bytes_v / bw(n, tier_v) + cpu_v / s_n

* parts_v, bytes_v, cpu_v  measured for the variant (results/calibration)
* l_req, bw                request latency and bandwidth between node n and the tier
* c_io                     parallel object-store requests per task
* s_n                      node speed relative to the calibration host core

Dynamic energy of a task: (P_max - P_idle) / k_n * e  (linear utilisation model).
Monetary cost of a task : price_n * e  (price per core-second; gateways are owned: 0).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TIERS = ("index", "warm", "summary")


@dataclass
class PlatformConfig:
    n_fog: int = 2
    k_fog: int = 4
    s_fog: float = 0.7
    n_cloud: int = 4
    k_cloud: int = 4
    s_cloud: float = 1.0
    s_gw: float = 0.2
    k_gw: int = 1
    # power (W) per node, linear model
    p_idle: dict = field(default_factory=lambda: {"gw": 2.7, "fog": 30.0, "cloud": 40.0})
    p_max: dict = field(default_factory=lambda: {"gw": 6.4, "fog": 90.0, "cloud": 120.0})
    # price per core-second (cost units; cloud = 0.048 per vCPU-hour)
    price: dict = field(default_factory=lambda: {"gw": 0.0, "fog": 0.024 / 3600, "cloud": 0.048 / 3600})
    # request latency (s) and bandwidth (bytes/s) from node type to tier
    wan_rtt: float = 0.120          # farm gateway <-> cloud (rural cellular)
    wan_bw: float = 1.25e6          # 10 Mbit/s
    fog_cloud_rtt: float = 0.040
    fog_cloud_bw: float = 12.5e6    # 100 Mbit/s
    cloud_store_lat: float = 0.020  # object-store request latency inside the cloud
    cloud_store_bw: float = 50e6
    local_lat: float = 0.001
    local_bw: float = 100e6
    c_io: int = 8
    # edge refresh (poll) model
    poll_latency: float = 1.5       # s per poll round (wake-up + radio round trip)
    poll_concurrency: int = 4       # stations polled in parallel per farm
    # execution-time noise (actual / estimated), lognormal sigma
    noise_sigma: float = 0.10

    def access(self, ntype: str, own_farm: bool, tier: str):
        """(request latency, bandwidth) from a node of type ntype to a data tier."""
        if tier == "index":           # latest-delivered-value index: on own gateway and in the cloud
            if ntype == "gw":
                return (self.local_lat, self.local_bw) if own_farm else (self.wan_rtt, self.wan_bw)
            if ntype == "fog":
                return self.fog_cloud_rtt, self.fog_cloud_bw
            return self.cloud_store_lat, self.cloud_store_bw
        if tier == "summary":         # summary tier: cloud store, replica on fog nodes
            if ntype == "gw":
                return self.wan_rtt, self.wan_bw
            if ntype == "fog":
                return self.local_lat, self.local_bw
            return self.cloud_store_lat, self.cloud_store_bw
        # warm tier: cloud object store only
        if ntype == "gw":
            return self.wan_rtt, self.wan_bw
        if ntype == "fog":
            return self.fog_cloud_rtt, self.fog_cloud_bw
        return self.cloud_store_lat, self.cloud_store_bw


@dataclass
class Variant:
    fid: str          # 'full' | 'summary' | 'std'
    tier: str
    parts: int
    nbytes: int
    cpu: float        # reference-core seconds
    error: float = 0.0


def exec_time(cfg: PlatformConfig, v: Variant, ntype: str, own_farm: bool, speed: float) -> float:
    lat, bw = cfg.access(ntype, own_farm, v.tier)
    rounds = int(np.ceil(v.parts / cfg.c_io)) if v.parts > 0 else 0
    return rounds * lat + v.nbytes / bw + v.cpu / speed


@dataclass
class Core:
    cid: int
    node: int
    ntype: str
    farm: int          # owning farm for gateways, -1 otherwise
    speed: float
    k: int             # cores of the node (for energy share)
    busy_until: float = 0.0
    running: object = None
    queue: list = field(default_factory=list)
    queued_work: float = 0.0
    busy_time: float = 0.0


def build_cores(cfg: PlatformConfig, n_farms: int) -> list[Core]:
    cores: list[Core] = []
    node = 0
    for _ in range(cfg.n_cloud):
        for _ in range(cfg.k_cloud):
            cores.append(Core(len(cores), node, "cloud", -1, cfg.s_cloud, cfg.k_cloud))
        node += 1
    for _ in range(cfg.n_fog):
        for _ in range(cfg.k_fog):
            cores.append(Core(len(cores), node, "fog", -1, cfg.s_fog, cfg.k_fog))
        node += 1
    for f in range(n_farms):
        for _ in range(cfg.k_gw):
            cores.append(Core(len(cores), node, "gw", f, cfg.s_gw, cfg.k_gw))
        node += 1
    return cores
