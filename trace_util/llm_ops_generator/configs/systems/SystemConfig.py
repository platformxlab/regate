from pydantic import BaseModel
from pydantic import TypeAdapter


# @dataclass(kw_only=True)
class SystemConfig(BaseModel):
    PUE: float
    carbon_intensity_kgCO2_per_kWh: float
