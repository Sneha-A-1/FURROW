"""Load config/default.yaml into the simulator dataclasses (single source of truth)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from ..simulator.platform import PlatformConfig
from ..simulator.run import ScenarioConfig

ROOT = Path(__file__).resolve().parents[3]


def load_default(path: Path | None = None) -> dict:
    return yaml.safe_load(open(path or ROOT / "config/default.yaml"))


def build(cfg: dict, overrides: dict | None = None):
    """Return (ScenarioConfig, PlatformConfig, furrow_kwargs) with flat overrides applied.

    Recognised override keys: any ScenarioConfig field name, any PlatformConfig field
    name, and 'furrow.<param>' for FURROW-S constructor arguments."""
    c = copy.deepcopy(cfg)
    sc, pl, fs = c["scenario"], c["platform"], c["scheduler"]["furrow"]
    scen = ScenarioConfig(
        n_farms=sc["n_farms"], window_s=sc["window_s"], sync_mode=sc["sync_mode"], rho=sc["rho"],
        deadline_u=sc["deadline_urgent_s"], deadline_s=sc["deadline_strategic_s"],
        connectivity=sc["connectivity"], budget_max=sc["budget_max"], budget_margin=sc["budget_margin_s"],
        shift_days=sc["shift_days"], season=tuple(sc["season"]), qcap=c["scheduler"]["local_queue_capacity"])
    plat = PlatformConfig(
        n_fog=pl["n_fog"], k_fog=pl["k_fog"], s_fog=pl["s_fog"], n_cloud=pl["n_cloud"], k_cloud=pl["k_cloud"],
        s_cloud=pl["s_cloud"], s_gw=pl["s_gw"],
        p_idle={k: v[0] for k, v in pl["power_w"].items()}, p_max={k: v[1] for k, v in pl["power_w"].items()},
        price={k: v / 3600.0 for k, v in pl["price_per_core_hour"].items()},
        wan_rtt=pl["wan"]["rtt_s"], wan_bw=float(pl["wan"]["bw_Bps"]),
        fog_cloud_rtt=pl["fog_cloud"]["rtt_s"], fog_cloud_bw=float(pl["fog_cloud"]["bw_Bps"]),
        cloud_store_lat=pl["cloud_store"]["lat_s"], cloud_store_bw=float(pl["cloud_store"]["bw_Bps"]),
        local_lat=pl["local"]["lat_s"], local_bw=float(pl["local"]["bw_Bps"]), c_io=pl["io_concurrency"],
        poll_latency=pl["poll"]["latency_s"], poll_concurrency=pl["poll"]["concurrency"],
        noise_sigma=pl["exec_noise_sigma"])
    fk = dict(r_max=fs["r_max"], tau_r=fs["tau_r_s"], guard=fs["guard_s"], gw_guard=fs["gw_guard_s"])
    for k, v in (overrides or {}).items():
        if k.startswith("furrow."):
            fk[k.split(".", 1)[1]] = v
        elif hasattr(scen, k):
            setattr(scen, k, type(getattr(scen, k))(v) if not isinstance(getattr(scen, k), tuple) else v)
        elif hasattr(plat, k):
            setattr(plat, k, type(getattr(plat, k))(v))
        else:
            raise KeyError(f"unknown override {k}")
    return scen, plat, fk
