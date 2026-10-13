# F1 source-table inventory

Source: Formula 1 World Championship (1950–2024), Kaggle dataset by Vopani, compiled from Ergast. The assignment-cited version is attached to the private Kaggle notebook. Audit snapshot: 2026-10-13.

The audit read all 14 CSVs with pandas keep_default_na=False. It counts the literal backslash-N sentinel and empty CSV fields as missing, without modifying source files. All listed candidate primary keys had 0 null-key rows and 0 duplicate-key rows in this snapshot. All foreign-key relationships below had 0 orphan rows after missing sentinel values were excluded. These are verified candidate relationships, not enforced database constraints.

## Table summary

| Table | Rows | Columns | Candidate primary key | Missing cells | FK orphan rows |
|---|---:|---:|---|---:|---:|
| circuits | 77 | 9 | circuitId | 0 | 0 |
| constructor_results | 12,625 | 5 | constructorResultsId | 12,608 | 0 |
| constructor_standings | 13,391 | 7 | constructorStandingsId | 0 | 0 |
| constructors | 212 | 5 | constructorId | 0 | 0 |
| driver_standings | 34,863 | 7 | driverStandingsId | 0 | 0 |
| drivers | 861 | 9 | driverId | 1,559 | 0 |
| lap_times | 589,081 | 6 | (raceId, driverId, lap) | 0 | 0 |
| pit_stops | 11,371 | 7 | (raceId, driverId, stop) | 0 | 0 |
| qualifying | 10,494 | 9 | qualifyId | 11,668 | 0 |
| races | 1,125 | 18 | raceId | 11,349 | 0 |
| results | 26,759 | 18 | resultId | 122,887 | 0 |
| seasons | 75 | 2 | year | 0 | 0 |
| sprint_results | 360 | 16 | resultId | 73 | 0 |
| status | 139 | 2 | statusId | 0 | 0 |

## Columns and missingness by table

| Table | Columns | Missing cells by column |
|---|---|---|
| circuits | circuitId, circuitRef, name, location, country, lat, lng, alt, url | None |
| constructor_results | constructorResultsId, raceId, constructorId, points, status | status: 12,608 |
| constructor_standings | constructorStandingsId, raceId, constructorId, points, position, positionText, wins | None |
| constructors | constructorId, constructorRef, name, nationality, url | None |
| driver_standings | driverStandingsId, raceId, driverId, points, position, positionText, wins | None |
| drivers | driverId, driverRef, number, code, forename, surname, dob, nationality, url | number: 802; code: 757 |
| lap_times | raceId, driverId, lap, position, time, milliseconds | None |
| pit_stops | raceId, driverId, stop, lap, time, duration, milliseconds | None |
| qualifying | qualifyId, raceId, driverId, constructorId, number, position, q1, q2, q3 | q1: 156; q2: 4,647; q3: 6,865 |
| races | raceId, year, round, circuitId, name, date, time, url, fp1_date, fp1_time, fp2_date, fp2_time, fp3_date, fp3_time, quali_date, quali_time, sprint_date, sprint_time | time: 731; fp1_date: 1,035; fp1_time: 1,057; fp2_date: 1,035; fp2_time: 1,057; fp3_date: 1,053; fp3_time: 1,072; quali_date: 1,035; quali_time: 1,057; sprint_date: 1,107; sprint_time: 1,110 |
| results | resultId, raceId, driverId, constructorId, number, grid, position, positionText, positionOrder, points, laps, time, milliseconds, fastestLap, rank, fastestLapTime, fastestLapSpeed, statusId | number: 6; position: 10,953; time: 19,079; milliseconds: 19,079; fastestLap: 18,507; rank: 18,249; fastestLapTime: 18,507; fastestLapSpeed: 18,507 |
| seasons | year, url | None |
| sprint_results | resultId, raceId, driverId, constructorId, number, grid, position, positionText, positionOrder, points, laps, time, milliseconds, fastestLap, fastestLapTime, statusId | position: 15; time: 20; milliseconds: 20; fastestLap: 9; fastestLapTime: 9 |
| status | statusId, status | None |

## Keys and relationships

- Single-column candidate keys: circuits.circuitId; constructors.constructorId; constructor_results.constructorResultsId; constructor_standings.constructorStandingsId; drivers.driverId; driver_standings.driverStandingsId; races.raceId; results.resultId; qualifying.qualifyId; sprint_results.resultId; seasons.year; status.statusId.
- Composite candidate keys: lap_times (raceId, driverId, lap); pit_stops (raceId, driverId, stop).
- races.circuitId → circuits.circuitId.
- constructor_results and constructor_standings: raceId → races.raceId; constructorId → constructors.constructorId.
- driver_standings: raceId → races.raceId; driverId → drivers.driverId.
- qualifying: raceId → races.raceId; driverId → drivers.driverId; constructorId → constructors.constructorId.
- results and sprint_results: raceId → races.raceId; driverId → drivers.driverId; constructorId → constructors.constructorId; statusId → status.statusId.
- lap_times and pit_stops: raceId → races.raceId; driverId → drivers.driverId.

## Sentinel and modeling notes

- The source uses literal backslash-N for unavailable values; this audit also counts empty fields. No other strings were normalized.
- Treat missingness by meaning: for example, Q2/Q3 may be absent because a driver did not advance; driver number/code are historically unavailable for many drivers. Do not impute without a feature-specific rationale.
- Parse dates and durations into typed working columns while keeping raw CSVs unchanged.
- Use only pre-race information as predictors. Race results, status, points, laps, pit stops, and post-race standings are leakage fields.
