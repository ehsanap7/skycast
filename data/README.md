# Local data

Dataset files in this folder are ignored by git. Only this README is tracked.

SeaweedFS is the active dataset store. This folder is the backup from the migration, and scheduled
jobs do not update it while `storage.backend` is `s3`.

```text
data/
  bronze/city=<slug>/   # original source files, never edited
  silver/city=<slug>/   # clean hourly Parquet
  gold/city=<slug>/     # model-ready Parquet and metadata
```

Place the raw CSV for a configured city in its bronze folder. City slugs come from
`config/cities.yaml`.
