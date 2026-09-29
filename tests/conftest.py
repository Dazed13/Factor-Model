"""Shared fixtures for Phase-1 ingestion tests."""

from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path

import polars as pl
import pytest


OLD_BHAV_CSV = """\
SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN
RELIANCE,EQ,2500.0,2550.0,2490.0,2520.0,2518.0,2480.0,1000000,2520000000.0,02-JAN-2020,50000,INE002A01018
TCS,EQ,2100.0,2120.0,2090.0,2110.0,2108.0,2080.0,500000,1055000000.0,02-JAN-2020,30000,INE467B01029
ILLIQUID,BZ,10.0,10.5,9.5,10.0,10.0,9.8,100,1000.0,02-JAN-2020,5,INEfakeBZ0001
DEBTCO,N1,100.0,100.0,100.0,100.0,100.0,100.0,0,0.0,02-JAN-2020,0,INEdebt000001
TRADETT,BE,50.0,52.0,49.0,51.0,51.0,50.0,2000,102000.0,02-JAN-2020,100,INEbe00000001
"""

UDIFF_BHAV_CSV = """\
TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd01,Rsvd02,Rsvd03,Rsvd04
2024-07-08,2024-07-08,CM,NSE,ST,,INE002A01018,RELIANCE,EQ,,,,,Reliance Industries,2500.0,2550.0,2490.0,2520.0,2518.0,2480.0,,,,,1000000,2520000000.0,50000,F1,1,,,,,
2024-07-08,2024-07-08,CM,NSE,ST,,INE467B01029,TCS,EQ,,,,,Tata Consultancy,2100.0,2120.0,2090.0,2110.0,2108.0,2080.0,,,,,500000,1055000000.0,30000,F1,1,,,,,
2024-07-08,2024-07-08,CM,NSE,ST,,INEfakeBZ0001,ILLIQUID,BZ,,,,,Illiquid Co,10.0,10.5,9.5,10.0,10.0,9.8,,,,,100,1000.0,5,F1,1,,,,,
2024-07-08,2024-07-08,CM,NSE,ST,,INEbe00000001,TRADETT,BE,,,,,Trade To Trade,50.0,52.0,49.0,51.0,51.0,50.0,,,,,2000,102000.0,100,F1,1,,,,,
"""


@pytest.fixture
def old_bhav_csv_bytes() -> bytes:
    return OLD_BHAV_CSV.encode("utf-8")


@pytest.fixture
def udiff_bhav_csv_bytes() -> bytes:
    return UDIFF_BHAV_CSV.encode("utf-8")


@pytest.fixture
def old_bhav_zip_bytes(old_bhav_csv_bytes: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("cm02JAN2020bhav.csv", old_bhav_csv_bytes)
    return buf.getvalue()


@pytest.fixture
def udiff_bhav_zip_bytes(udiff_bhav_csv_bytes: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("BhavCopy_NSE_CM_0_0_0_20240708_F_0000.csv", udiff_bhav_csv_bytes)
    return buf.getvalue()


@pytest.fixture
def fixtures_dir(tmp_path: Path, old_bhav_zip_bytes: bytes, udiff_bhav_zip_bytes: bytes) -> Path:
    (tmp_path / "cm02JAN2020bhav.csv.zip").write_bytes(old_bhav_zip_bytes)
    (tmp_path / "BhavCopy_NSE_CM_0_0_0_20240708_F_0000.csv.zip").write_bytes(
        udiff_bhav_zip_bytes
    )
    return tmp_path


@pytest.fixture
def sample_universe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "as_of_date": [date(2020, 1, 1), date(2020, 1, 1), date(2021, 6, 1)],
            "symbol": ["RELIANCE", "TCS", "RELIANCE"],
            "company_name": ["Reliance", "TCS", "Reliance"],
            "industry": ["Energy", "IT", "Energy"],
            "isin": ["INE002A01018", "INE467B01029", "INE002A01018"],
        }
    )
