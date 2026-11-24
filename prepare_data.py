import argparse
import math
import re
import sqlite3
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


WEATHER_COLUMNS = [
    "TEMP", "DEWP", "SLP", "STP", "VISIB", "WDSP", "MXSPD",
    "MAX", "MIN", "PRCP", "RH",
]

EXTREME_COLUMNS = [
    "HIGH_TEMPERATURE", "LOW_TEMPERATURE", "HIGH_HUMIDITY",
    "HEAT_INDEX_CAUTION", "HEAT_INDEX_EXTREME_CAUTION",
    "HEAT_INDEX_DANGER", "HEAT_INDEX_EXTREME_DANGER",
    "WIND_CHILL_VERY_COLD", "WIND_CHILL_FROSTBITE_DANGER",
    "WIND_CHILL_GREAT_FROSTBITE_DANGER",
    "WIND_LEVEL_0", "WIND_LEVEL_1", "WIND_LEVEL_2",
    "WIND_LEVEL_3", "WIND_LEVEL_4", "WIND_LEVEL_5",
    "WIND_LEVEL_6", "WIND_LEVEL_7", "WIND_LEVEL_8",
    "WIND_LEVEL_9", "WIND_LEVEL_10", "WIND_LEVEL_11",
    "WIND_LEVEL_12", "PRECIPITATION_50", "PRECIPITATION_100",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare transformer load data from Transformer_DB.db for CMPR-Mixer."
    )
    parser.add_argument(
        "--db_path",
        type=Path,
        default=SCRIPT_DIR / "data" / "raw" / "Transformer_DB.db",
        help="Path to Transformer_DB.db.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "transformer",
        help="Directory for generated CSV files.",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="Time frequency used to rebuild the regular time index.",
    )
    parser.add_argument(
        "--max_missing_rate",
        type=float,
        default=1.0,
        help=(
            "Drop active transformers whose LOAD missing rate is larger than this value. "
            "Default 1.0 keeps all non-deleted transformers."
        ),
    )
    parser.add_argument(
        "--split_group_files",
        action="store_true",
        help="Also write one CSV file for each city and each community.",
    )
    parser.add_argument(
        "--skip_exog",
        action="store_true",
        help="Do not generate exogenous-variable CSV files.",
    )
    return parser.parse_args()


def load_active_transformer_data(db_path):
    query = """
        SELECT
            r.DATETIME AS date,
            r.TRANSFORMER_ID AS transformer_id,
            r.LOAD AS load,
            m.CITY AS city,
            m.DISTRICT AS community,
            m.YXRL AS capacity,
            m.CLOSEST_STATION AS closest_station
        FROM transformer_raw r
        INNER JOIN transformer_meta m
            ON r.TRANSFORMER_ID = m.TRANSFORMER_ID
        WHERE m."DELETE" IS NULL OR m."DELETE" = 0
        ORDER BY r.TRANSFORMER_ID, r.DATETIME
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    df["date"] = pd.to_datetime(df["date"])
    df["load"] = pd.to_numeric(df["load"], errors="coerce")
    df["city"] = pd.to_numeric(df["city"], errors="coerce").astype("Int64")
    df["community"] = pd.to_numeric(df["community"], errors="coerce").astype("Int64")
    return df


def build_quality_table(df):
    quality = (
        df.groupby(["transformer_id", "city", "community"], dropna=False)
        .agg(
            rows=("load", "size"),
            missing_loads=("load", lambda x: int(x.isna().sum())),
            valid_loads=("load", lambda x: int(x.notna().sum())),
            min_load=("load", "min"),
            max_load=("load", "max"),
            mean_load=("load", "mean"),
        )
        .reset_index()
    )
    quality["missing_rate"] = quality["missing_loads"] / quality["rows"]
    return quality.sort_values(["missing_rate", "transformer_id"], ascending=[False, True])


def interpolate_load_matrix(df, quality, max_missing_rate, freq):
    keep_ids = quality.loc[
        quality["missing_rate"] <= max_missing_rate, "transformer_id"
    ].tolist()
    if not keep_ids:
        raise ValueError("No transformer remains after max_missing_rate filtering.")

    df = df[df["transformer_id"].isin(keep_ids)].copy()

    start_time = df["date"].min()
    end_time = df["date"].max()
    full_index = pd.date_range(start=start_time, end=end_time, freq=freq)

    load_wide = (
        df.pivot(index="date", columns="transformer_id", values="load")
        .sort_index()
        .reindex(full_index)
    )
    load_wide.index.name = "date"


    load_wide = load_wide.interpolate(method="time", limit_direction="both")
    load_wide = load_wide.ffill().bfill()
    return load_wide, df


def safe_feature_name(prefix, value):
    text = "" if value is None else str(value).strip()
    if text == "" or text.lower() in {"none", "nan", "null"}:
        return None
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text.lower()).strip("_")
    if not text:
        return None
    return f"{prefix}_{text}"


def normalize_station_id(value):
    """Return one stable text representation for SQLite/pandas station ids."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def load_station_metadata(conn):
    station_meta = pd.read_sql_query(
        """
        SELECT
            STATION_ID AS station_id,
            "STATION NAME" AS station_name,
            LAT AS latitude,
            LON AS longitude
        FROM weather_meta
        """,
        conn,
    )
    station_meta["station_id"] = station_meta["station_id"].map(
        normalize_station_id
    )
    station_meta["latitude"] = pd.to_numeric(
        station_meta["latitude"], errors="coerce"
    )
    station_meta["longitude"] = pd.to_numeric(
        station_meta["longitude"], errors="coerce"
    )
    return station_meta.drop_duplicates("station_id").set_index("station_id")


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance used only to order fallback weather stations."""
    radius_km = 6371.0088
    lat1, lon1, lat2, lon2 = map(
        math.radians, (lat1, lon1, lat2, lon2)
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(math.sqrt(value))


def nearest_station_order(target_station, station_ids, station_meta):
    """Order candidate fallback stations from geographically near to far."""
    candidates = [station for station in station_ids if station != target_station]
    if target_station not in station_meta.index:
        return sorted(candidates)

    target = station_meta.loc[target_station]
    if pd.isna(target["latitude"]) or pd.isna(target["longitude"]):
        return sorted(candidates)

    def distance(station):
        if station not in station_meta.index:
            return float("inf")
        candidate = station_meta.loc[station]
        if pd.isna(candidate["latitude"]) or pd.isna(candidate["longitude"]):
            return float("inf")
        return haversine_km(
            target["latitude"],
            target["longitude"],
            candidate["latitude"],
            candidate["longitude"],
        )

    return sorted(candidates, key=lambda station: (distance(station), station))


def build_station_feature_frames(
    df,
    value_columns,
    daily_index,
    station_meta,
    prefix,
    aggregation="mean",
    empty_values_are_zero=False,
):
    """Build one complete daily feature frame for every weather station.

    Missing cells first use the geographically nearest station that has a
    value on the same day. Remaining continuous-weather gaps are then filled
    by time interpolation. Extreme-weather indicators use zero only after the
    same-day nearest-station fallback has been exhausted.
    """
    if df.empty:
        return {}

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["station_id"] = df["station_id"].map(normalize_station_id)
    for column in value_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if empty_values_are_zero:
            df[column] = df[column].fillna(0.0)

    grouped = df.groupby(["date", "station_id"], dropna=False)[value_columns]
    if aggregation == "max":
        grouped = grouped.max()
    else:
        grouped = grouped.mean()

    station_ids = sorted(df["station_id"].dropna().unique())
    raw_frames = {}
    for station in station_ids:
        try:
            frame = grouped.xs(station, level="station_id")
        except KeyError:
            frame = pd.DataFrame(columns=value_columns)
        raw_frames[station] = frame.reindex(daily_index)

    completed = {}
    for station in station_ids:
        frame = raw_frames[station].copy()
        for fallback in nearest_station_order(
            station, station_ids, station_meta
        ):
            frame = frame.fillna(raw_frames[fallback])

        if empty_values_are_zero:
            frame = frame.fillna(0.0)
        else:
            frame = frame.interpolate(method="time", limit_direction="both")
            frame = frame.ffill().bfill().fillna(0.0)

        frame.columns = [f"{prefix}_{column.lower()}" for column in value_columns]
        completed[station] = frame.astype(float)
    return completed


def average_station_frames(station_frames, station_ids=None):
    """Equal arithmetic mean; transformer capacity is never used."""
    if station_ids is None:
        station_ids = sorted(station_frames)
    else:
        station_ids = [
            normalize_station_id(station)
            for station in station_ids
            if normalize_station_id(station) in station_frames
        ]
        station_ids = sorted(set(station_ids))
    if not station_ids:
        raise ValueError("No matching weather station is available.")

    result = station_frames[station_ids[0]].copy()
    for station in station_ids[1:]:
        result = result.add(station_frames[station], fill_value=0.0)
    return result / float(len(station_ids))


def load_weather_exog(conn, daily_index, station_meta):

    weather = pd.read_sql_query(
        """
        SELECT
            DATETIME AS date,
            STATION_ID AS station_id,
            TEMP, DEWP, SLP, STP, VISIB, WDSP, MXSPD,
            MAX, MIN, PRCP, RH
        FROM weather
        """,
        conn,
    )
    weather_start = pd.to_datetime(weather["date"]).min().normalize()
    weather_end = pd.to_datetime(weather["date"]).max().normalize()
    station_frames = build_station_feature_frames(
        weather,
        WEATHER_COLUMNS,
        daily_index,
        station_meta,
        prefix="weather",
        aggregation="mean",
        empty_values_are_zero=False,
    )
    return (
        average_station_frames(station_frames),
        station_frames,
        weather_start,
        weather_end,
    )


def load_extreme_calculated_exog(conn, daily_index, station_meta):
    extreme = pd.read_sql_query(
        """
        SELECT
            DATETIME AS date,
            STATION_ID AS station_id,
            HIGH_TEMPERATURE, LOW_TEMPERATURE, HIGH_HUMIDITY,
            HEAT_INDEX_CAUTION, HEAT_INDEX_EXTREME_CAUTION,
            HEAT_INDEX_DANGER, HEAT_INDEX_EXTREME_DANGER,
            WIND_CHILL_VERY_COLD, WIND_CHILL_FROSTBITE_DANGER,
            WIND_CHILL_GREAT_FROSTBITE_DANGER,
            WIND_LEVEL_0, WIND_LEVEL_1, WIND_LEVEL_2,
            WIND_LEVEL_3, WIND_LEVEL_4, WIND_LEVEL_5,
            WIND_LEVEL_6, WIND_LEVEL_7, WIND_LEVEL_8,
            WIND_LEVEL_9, WIND_LEVEL_10, WIND_LEVEL_11,
            WIND_LEVEL_12, PRECIPITATION_50, PRECIPITATION_100
        FROM extreme_weather_calculated
        """,
        conn,
    )
    station_frames = build_station_feature_frames(
        extreme,
        EXTREME_COLUMNS,
        daily_index,
        station_meta,
        prefix="extreme",
        aggregation="max",
        empty_values_are_zero=True,
    )
    return average_station_frames(station_frames), station_frames


def load_holiday_exog(conn, daily_index):
    holiday = pd.read_sql_query(
        "SELECT DATETIME AS date, HOLIDAY AS holiday FROM holiday",
        conn,
    )
    holiday["date"] = pd.to_datetime(holiday["date"]).dt.normalize()
    holiday["holiday"] = holiday["holiday"].fillna("none")

    out = pd.DataFrame(index=daily_index)
    is_holiday = (
        holiday.assign(is_holiday=lambda x: x["holiday"].str.lower() != "none")
        .groupby("date")["is_holiday"]
        .max()
        .reindex(daily_index)
        .fillna(False)
        .astype(float)
    )
    out["is_holiday"] = is_holiday

    for value in sorted(holiday["holiday"].dropna().unique()):
        col_name = safe_feature_name("holiday", value)
        if col_name is None:
            continue
        out[col_name] = (
            (holiday["holiday"] == value).astype(float)
            .groupby(holiday["date"])
            .max()
            .reindex(daily_index)
            .fillna(0.0)
        )

    return out


def load_internet_extreme_exog(conn, daily_index):
    hazard = pd.read_sql_query(
        "SELECT DATETIME AS date, HAZARD AS hazard FROM extreme_weather_internet",
        conn,
    )
    hazard["date"] = pd.to_datetime(hazard["date"]).dt.normalize()
    hazard["hazard"] = hazard["hazard"].fillna("none")

    out = pd.DataFrame(index=daily_index)
    out["is_internet_extreme"] = (
        hazard.assign(is_extreme=lambda x: x["hazard"].str.lower() != "none")
        .groupby("date")["is_extreme"]
        .max()
        .reindex(daily_index)
        .fillna(False)
        .astype(float)
    )

    for value in sorted(hazard["hazard"].dropna().unique()):
        col_name = safe_feature_name("internet_extreme", value)
        if col_name is None:
            continue
        out[col_name] = (
            (hazard["hazard"] == value).astype(float)
            .groupby(hazard["date"])
            .max()
            .reindex(daily_index)
            .fillna(0.0)
        )

    return out


def expand_daily_features_to_hourly(daily_features, hourly_index):
    hourly = pd.DataFrame({"date": hourly_index})
    hourly["day"] = hourly["date"].dt.normalize()
    return hourly.merge(
        daily_features.reset_index(),
        on="day",
        how="left",
    ).drop(columns=["day"])


def build_and_write_exogenous_files(load_wide, meta_df, db_path, output_dir):
    load_start = load_wide.index.min().normalize()
    load_end = load_wide.index.max().normalize()

    with sqlite3.connect(db_path) as conn:
        raw_weather_dates = pd.read_sql_query(
            "SELECT MIN(DATETIME) AS min_date, MAX(DATETIME) AS max_date FROM weather",
            conn,
        )
        weather_start = pd.to_datetime(raw_weather_dates.loc[0, "min_date"]).normalize()
        weather_end = pd.to_datetime(raw_weather_dates.loc[0, "max_date"]).normalize()

        exog_start = max(load_start, weather_start)
        exog_end = min(load_end, weather_end)
        daily_index = pd.date_range(exog_start, exog_end, freq="D")

        station_meta = load_station_metadata(conn)
        (
            weather_exog,
            weather_station_frames,
            _,
            _,
        ) = load_weather_exog(conn, daily_index, station_meta)
        (
            extreme_exog,
            extreme_station_frames,
        ) = load_extreme_calculated_exog(
            conn, daily_index, station_meta
        )
        holiday_exog = load_holiday_exog(conn, daily_index)
        internet_extreme_exog = load_internet_extreme_exog(conn, daily_index)

    exog_daily = pd.concat(
        [weather_exog, extreme_exog, holiday_exog, internet_extreme_exog],
        axis=1,
    )
    exog_daily = exog_daily.loc[:, ~exog_daily.columns.duplicated()]
    exog_daily = exog_daily.fillna(0.0)
    exog_daily.index.name = "day"

    hourly_index = load_wide.loc[
        (load_wide.index >= exog_start)
        & (load_wide.index <= exog_end + pd.Timedelta(hours=23))
    ].index
    exog_hourly = expand_daily_features_to_hourly(exog_daily, hourly_index)

    total_load = load_wide.sum(axis=1).rename("load")
    load_df = total_load.loc[hourly_index].reset_index()
    load_df.columns = ["date", "load"]

    load_exog = exog_hourly.merge(load_df, on="date", how="inner")
    exog_hourly.to_csv(output_dir / "transformer_exogenous_hourly.csv", index=False)
    load_exog.to_csv(output_dir / "transformer_load_all_exog.csv", index=False)


    meta = (
        meta_df[["transformer_id", "city", "closest_station"]]
        .drop_duplicates("transformer_id")
        .set_index("transformer_id")
        .loc[load_wide.columns]
    )
    city_records = []
    for city, group_meta in meta.groupby("city", dropna=False):
        city_id = int(city)
        station_ids = sorted(
            {
                normalize_station_id(station)
                for station in group_meta["closest_station"].dropna()
            }
        )
        city_weather = average_station_frames(
            weather_station_frames, station_ids
        )
        city_extreme = average_station_frames(
            extreme_station_frames, station_ids
        )
        city_exog_daily = pd.concat(
            [
                city_weather,
                city_extreme,
                holiday_exog,
                internet_extreme_exog,
            ],
            axis=1,
        )
        city_exog_daily = city_exog_daily.loc[
            :, ~city_exog_daily.columns.duplicated()
        ].fillna(0.0)
        city_exog_daily.index.name = "day"
        city_exog_hourly = expand_daily_features_to_hourly(
            city_exog_daily, hourly_index
        )

        city_load = load_wide[group_meta.index].sum(axis=1).loc[hourly_index]
        city_load_df = city_load.rename("load").reset_index()
        city_load_df.columns = ["date", "load"]
        city_exog = city_exog_hourly.merge(
            city_load_df, on="date", how="inner"
        )
        city_file = f"transformer_load_city_{city_id}_exog.csv"
        city_exog.to_csv(output_dir / city_file, index=False)

        station_names = [
            str(station_meta.loc[station, "station_name"])
            if station in station_meta.index
            else "unknown"
            for station in station_ids
        ]
        city_records.append(
            {
                "city": city_id,
                "data_path": city_file,
                "transformers": len(group_meta),
                "rows": len(city_exog),
                "closest_stations": "|".join(station_ids),
                "station_names": "|".join(station_names),
            }
        )

    city_manifest_file = "transformer_city_exog_files.csv"
    pd.DataFrame(city_records).sort_values("city").to_csv(
        output_dir / city_manifest_file,
        index=False,
    )

    exog_columns = [c for c in load_exog.columns if c not in {"date", "load"}]
    with (output_dir / "transformer_exog_columns.txt").open("w", encoding="utf-8") as f:
        for idx, col in enumerate(exog_columns):
            f.write(f"{idx},{col}\n")

    return {
        "exog_file": "transformer_load_all_exog.csv",
        "exog_only_file": "transformer_exogenous_hourly.csv",
        "exog_columns_file": "transformer_exog_columns.txt",
        "exog_rows": len(load_exog),
        "exog_features": len(exog_columns),
        "weather_features": len(WEATHER_COLUMNS),
        "extreme_features": len(EXTREME_COLUMNS),
        "exog_start_time": str(load_exog["date"].min()),
        "exog_end_time": str(load_exog["date"].max()),
        "city_exog_manifest": city_manifest_file,
        "city_exog_files": len(city_records),
        "weather_mode_total": "equal_mean_all_stations",
        "weather_mode_city": "closest_station",
        "removed_weather_features": "GUST,SNDP",
    }


def write_single_target_csv(series, path):
    out = series.rename("load").reset_index()
    out.columns = ["date", "load"]
    out.to_csv(path, index=False)


def write_wide_csv(df, path):
    out = df.reset_index()
    out = out.rename(columns={"index": "date"})
    out.to_csv(path, index=False)


def aggregate_and_write(load_wide, meta_df, output_dir, split_group_files):
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = (
        meta_df[["transformer_id", "city", "community"]]
        .drop_duplicates("transformer_id")
        .set_index("transformer_id")
        .loc[load_wide.columns]
    )

    total_load = load_wide.sum(axis=1)
    write_single_target_csv(total_load, output_dir / "transformer_load_all.csv")

    city_frames = {}
    for city, group_meta in meta.groupby("city", dropna=False):
        col_name = f"city_{int(city)}"
        city_load = load_wide[group_meta.index].sum(axis=1)
        city_frames[col_name] = city_load
        if split_group_files:
            write_single_target_csv(city_load, output_dir / f"transformer_load_{col_name}.csv")

    city_wide = pd.DataFrame(city_frames, index=load_wide.index).sort_index(axis=1)
    write_wide_csv(city_wide, output_dir / "transformer_load_by_city.csv")

    community_frames = {}
    for (city, community), group_meta in meta.groupby(["city", "community"], dropna=False):
        col_name = f"city_{int(city)}_community_{int(community)}"
        community_load = load_wide[group_meta.index].sum(axis=1)
        community_frames[col_name] = community_load
        if split_group_files:
            write_single_target_csv(
                community_load,
                output_dir / f"transformer_load_{col_name}.csv",
            )

    community_wide = pd.DataFrame(community_frames, index=load_wide.index).sort_index(axis=1)
    write_wide_csv(community_wide, output_dir / "transformer_load_by_community.csv")

    return {
        "total_file": "transformer_load_all.csv",
        "city_file": "transformer_load_by_city.csv",
        "community_file": "transformer_load_by_community.csv",
        "rows": len(load_wide),
        "transformers": len(load_wide.columns),
        "cities": len(city_frames),
        "communities": len(community_frames),
        "start_time": str(load_wide.index.min()),
        "end_time": str(load_wide.index.max()),
    }


def write_summary(summary, quality, output_dir, max_missing_rate, exog_summary=None):
    summary_path = output_dir / "preprocess_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Transformer load preprocessing summary\n")
        f.write(f"start_time: {summary['start_time']}\n")
        f.write(f"end_time: {summary['end_time']}\n")
        f.write(f"rows: {summary['rows']}\n")
        f.write(f"kept_transformers: {summary['transformers']}\n")
        f.write(f"cities: {summary['cities']}\n")
        f.write(f"communities: {summary['communities']}\n")
        f.write(f"max_missing_rate: {max_missing_rate}\n")
        f.write(f"total_file: {summary['total_file']}\n")
        f.write(f"city_file: {summary['city_file']}\n")
        f.write(f"community_file: {summary['community_file']}\n")
        if exog_summary is not None:
            f.write(f"exog_file: {exog_summary['exog_file']}\n")
            f.write(f"exog_only_file: {exog_summary['exog_only_file']}\n")
            f.write(f"exog_columns_file: {exog_summary['exog_columns_file']}\n")
            f.write(f"exog_rows: {exog_summary['exog_rows']}\n")
            f.write(f"exog_features: {exog_summary['exog_features']}\n")
            f.write(
                f"weather_features: {exog_summary['weather_features']}\n"
            )
            f.write(
                f"extreme_features: {exog_summary['extreme_features']}\n"
            )
            f.write(f"exog_start_time: {exog_summary['exog_start_time']}\n")
            f.write(f"exog_end_time: {exog_summary['exog_end_time']}\n")
            f.write(f"city_exog_manifest: {exog_summary['city_exog_manifest']}\n")
            f.write(f"city_exog_files: {exog_summary['city_exog_files']}\n")
            f.write(
                f"weather_mode_total: {exog_summary['weather_mode_total']}\n"
            )
            f.write(
                f"weather_mode_city: {exog_summary['weather_mode_city']}\n"
            )
            f.write(
                "removed_weather_features: "
                f"{exog_summary['removed_weather_features']}\n"
            )
        f.write("\nNote: missing LOAD values are filled per transformer by time interpolation.\n")
        if exog_summary is not None:
            f.write(
                "Note: total-load weather/extreme features use an equal mean "
                "over stations; each city uses its CLOSEST_STATION features. "
                "Missing station values use the nearest available station on "
                "the same day before interpolation. Transformer capacity is "
                "never used.\n"
            )

    quality.to_csv(output_dir / "transformer_quality.csv", index=False)


def main():
    args = parse_args()
    print(f"Reading database: {args.db_path}")
    df = load_active_transformer_data(args.db_path)

    print("Building transformer quality table...")
    quality = build_quality_table(df)
    raw_transformers = quality["transformer_id"].nunique()
    kept_transformers = int((quality["missing_rate"] <= args.max_missing_rate).sum())
    print(f"Active transformers: {raw_transformers}, kept: {kept_transformers}")

    print("Interpolating missing LOAD values per transformer...")
    load_wide, kept_df = interpolate_load_matrix(
        df=df,
        quality=quality,
        max_missing_rate=args.max_missing_rate,
        freq=args.freq,
    )

    print(f"Writing processed CSV files to: {args.output_dir}")
    summary = aggregate_and_write(
        load_wide=load_wide,
        meta_df=kept_df,
        output_dir=args.output_dir,
        split_group_files=args.split_group_files,
    )
    exog_summary = None
    if not args.skip_exog:
        print("Building hourly exogenous-variable CSV files...")
        exog_summary = build_and_write_exogenous_files(
            load_wide=load_wide,
            meta_df=kept_df,
            db_path=args.db_path,
            output_dir=args.output_dir,
        )
    write_summary(summary, quality, args.output_dir, args.max_missing_rate, exog_summary)

    print("Done.")
    print(f"Total load: {args.output_dir / summary['total_file']}")
    print(f"By city: {args.output_dir / summary['city_file']}")
    print(f"By community: {args.output_dir / summary['community_file']}")
    print(f"Quality table: {args.output_dir / 'transformer_quality.csv'}")
    if exog_summary is not None:
        print(f"Load + exogenous variables: {args.output_dir / exog_summary['exog_file']}")
        print(
            "City load + exogenous manifest: "
            f"{args.output_dir / exog_summary['city_exog_manifest']}"
        )
        print(f"Exogenous column list: {args.output_dir / exog_summary['exog_columns_file']}")


if __name__ == "__main__":
    main()
