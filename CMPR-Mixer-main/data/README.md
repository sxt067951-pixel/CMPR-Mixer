# Data specification

Data source: [published database on Figshare](https://doi.org/10.6084/m9.figshare.26333452.v1).
Place `Transformer_DB.db` in `data/raw/`, then run `python prepare_data.py` from the repository root.

## Required prepared files

The default directory is `data/transformer/`; `--data_root` may point elsewhere.
Forecasting can use one selected file. Transfer entry points validate that all ten
city files are present and share the same header.

- Total: `transformer_load_all_exog.csv`.
- Cities: `transformer_load_city_0_exog.csv` through `transformer_load_city_9_exog.csv`.
- CSV order: `date`, the following 56 features, then `load`.
- One strictly increasing, unique timestamp per hour; numeric finite observations.

Feature positions below are **zero-based within the 56 exogenous columns**, not
including `date`. The loader standardizes these columns; the retrieval module uses
their positions, so keep their order unchanged.

| Positions | Contents | Retrieval encoding |
| --- | --- | --- |
| 0–10 | `weather_temp,dewp,slp,stp,visib,wdsp,mxspd,max,min,prcp,rh` with `weather_` prefix | Numeric MLP |
| 11–35 | 25 calculated extreme-weather indicators with `extreme_` prefix | Same numeric MLP |
| 36 | `is_holiday` | Input channel; not separately added to the context sum |
| 37–43 | Seven holiday-type indicator columns | Eight-state embedding, including none |
| 44 | `is_internet_extreme` | Input channel; not separately added to the context sum |
| 45–55 | Eleven internet-event indicator columns | Twelve-state embedding, including none |

Exact calculated extreme-weather order:

```text
extreme_high_temperature
extreme_low_temperature
extreme_high_humidity
extreme_heat_index_caution
extreme_heat_index_extreme_caution
extreme_heat_index_danger
extreme_heat_index_extreme_danger
extreme_wind_chill_very_cold
extreme_wind_chill_frostbite_danger
extreme_wind_chill_great_frostbite_danger
extreme_wind_level_0
extreme_wind_level_1
extreme_wind_level_2
extreme_wind_level_3
extreme_wind_level_4
extreme_wind_level_5
extreme_wind_level_6
extreme_wind_level_7
extreme_wind_level_8
extreme_wind_level_9
extreme_wind_level_10
extreme_wind_level_11
extreme_wind_level_12
extreme_precipitation_50
extreme_precipitation_100
```

Holiday suffix order after `holiday_`:
`dragon_boat_festival, international_labor_day, mid_autumn_festival, national_day,
new_year, qingming, spring_festival`.

Internet-event suffix order after `internet_extreme_`:
`cold_wave, dragon_boat_rain, drought, excessive_flooding, extreme_rainfall,
hot_weather, rainstorm, severe_convective_weather, tropical_storm_sanba,
typhoon_chaba, typhoon_haikui`.

Type IDs preserve the earlier implementation: first positive standardized indicator
wins; no positive entry gives ID zero. This is not a new multilabel encoding, and
overlapping categories are not independently summed. Pooling and standardization
can affect these IDs, as in the original implementation.

## Windows and scaling

Use the first 70% of rows for training and last 20% for testing; the remainder is
validation. `StandardScaler` is fit on the training rows only and reused for the
same city's validation/test rows. Validation and test inputs may borrow the
preceding history; their forecast labels stay within their own splits.

The inspected prepared files contain 16,296 hourly observations spanning
2022-01-01 00:00 to 2023-11-10 23:00. With `168 → 24`, the train/validation/test
window counts are 11,216 / 1,607 / 3,236. Training drops a final incomplete batch;
evaluation includes all windows. These are observations about the current local
files, not enforced fixed dates for every future dataset.

Calendar marks are hour, weekday, day of month, and day of year, mapped to roughly
`[-0.5, 0.5]`. They are generated from timestamps, not additional downloaded data.
The context branch uses hour and weekday; the value/time embedding uses all four.
Future calendar features are off by default. Future observed weather is never
supplied to the forecast forward pass.

## Rebuilding from SQLite

```bash
python prepare_data.py --db_path data/raw/Transformer_DB.db --output_dir data/transformer
```

The preprocessor expects the original transformer, station, weather, calculated
weather-event, holiday, and internet-event table schemas. It is not a generic
SQLite-to-CSV converter. Existing prepared files can instead be used directly.

The retained procedure:

1. Select non-deleted transformers, build a
   regular hourly index. Keep the original missing-rate rule and time
   interpolation/forward-backward filling.
2. Sum transformer load into total, city, and community series. Capacity is **not**
   used to weight load or external features.
3. Use an equal mean across weather stations for total load. For a city, use the
   distinct `CLOSEST_STATION` IDs of its transformers, equally averaging when more
   than one station is associated with that city.
4. Fill station missing values from the nearest available station on the same day,
   then retain the original temporal interpolation/fill fallback. Use the eleven weather columns listed above.
5. Align to the overlap of load/weather coverage. Expand daily weather and event
   features to hourly rows, so a day's original daily value is repeated for its
   24 hours. Export total/city CSVs and preprocessing metadata.

The original offline preprocessing can use observations on both sides of a gap.
It is therefore not a strictly causal online imputation protocol. 

## Transfer interpretation

Every city retains its own scaler fit on its complete training-period rows,
including target cities. Fine-tuning ratios select nested final portions of
**training windows**, not independent rows. Even at 0% adaptation, target-domain
normalization statistics are available. Also, the entire target validation split
is used to select fine-tuned checkpoints.
