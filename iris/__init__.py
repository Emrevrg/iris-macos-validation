"""IRIS — kayıpsız dağıtık model motoru. Herhangi bir modeli, bellek bütçesi altında,
bit-birebir çalıştırır (bit sıkıştırma değil; kayıpsız katman-akışı/dağıtım)."""
from iris.engine import IrisModel, verify_bit_exact, shard_plan, HubSource, LocalSource  # noqa: F401
__version__ = "0.1.0"
__all__ = ["IrisModel", "verify_bit_exact", "shard_plan", "HubSource", "LocalSource"]
