"""Run the data layers from bronze to gold."""

from dataclasses import dataclass

from skycast.config import Settings
from skycast.data.bronze import load_bronze
from skycast.data.contracts import GOLD_DATA_FILE
from skycast.data.gold import build_gold, write_gold
from skycast.data.silver import build_silver, read_silver, write_silver
from skycast.data.storage import Layer, build_store, dataset_key


@dataclass(frozen=True, slots=True)
class PipelineSummary:
    """What a pipeline run produced."""

    bronze_rows: int
    silver_rows: int
    gold_rows: int
    gold_key: str


def run_data_pipeline(settings: Settings) -> PipelineSummary:
    """Build silver and gold for the active city."""
    store = build_store(settings.storage)
    city = settings.active_city
    bronze = load_bronze(store, city)
    silver = build_silver(bronze, city)
    write_silver(store, city, silver)
    gold = build_gold(read_silver(store, city), city, settings.gold, settings.horizon_days)
    write_gold(store, city, gold, settings.gold, settings.horizon_days)
    return PipelineSummary(
        bronze_rows=len(bronze),
        silver_rows=len(silver.frame),
        gold_rows=len(gold.frame),
        gold_key=dataset_key(Layer.GOLD, city.slug, GOLD_DATA_FILE),
    )
