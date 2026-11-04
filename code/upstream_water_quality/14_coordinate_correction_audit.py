# -*- coding: utf-8 -*-
"""Create a before/after audit table for nine repaired river sections."""
from pathlib import Path
import pandas as pd
from pyproj import Geod

OUT = Path(__file__).resolve().parent.parent
PROC, QA = OUT / "01_INPUT与过程数据", OUT / "06_QA"
IDS = [
    "SZ_WQ_RIV_001", "SZ_WQ_RIV_003", "SZ_WQ_RIV_004",
    "SZ_WQ_RIV_005", "SZ_WQ_RIV_006", "SZ_WQ_RIV_009",
    "SZ_WQ_RIV_010", "SZ_WQ_RIV_011", "SZ_WQ_RIV_016",
]

old = pd.read_csv(PROC / "WQ_Site_Special_13Ranges.csv", low_memory=False)
old = old[old.station_id.isin(IDS)][["station_id", "station_name_canonical", "longitude", "latitude", "spatial_id"]]
old = old.rename(columns={"station_name_canonical":"station_name_before", "longitude":"longitude_before",
                          "latitude":"latitude_before", "spatial_id":"geometry_id_before"})
reg = pd.read_csv(PROC / "WQ_Station_Spatial_Registry.csv", low_memory=False)
new = reg[reg.station_id.isin(IDS)][["station_id", "station_name_canonical", "waterbody_name_spatial",
                                    "provided_section_longitude", "provided_section_latitude",
                                    "spatial_longitude", "spatial_latitude", "geometry_id",
                                    "river_snap_distance_m", "spatial_quality"]]
new = new.rename(columns={"station_name_canonical":"station_name_after", "geometry_id":"geometry_id_after",
                          "spatial_longitude":"longitude_after", "spatial_latitude":"latitude_after"})
d = old.merge(new, on="station_id", validate="1:1")
geod = Geod(ellps="WGS84")
d["old_to_new_shift_m"] = [geod.inv(a,b,c,e)[2] for a,b,c,e in zip(
    d.longitude_before, d.latitude_before, d.longitude_after, d.latitude_after)]
d.to_csv(QA / "9个River具名断面坐标修缮前后审计.csv", index=False, encoding="utf-8-sig")
print(d[["station_id","station_name_after","longitude_after","latitude_after","old_to_new_shift_m"]].to_string(index=False))
