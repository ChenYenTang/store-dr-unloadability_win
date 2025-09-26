
from typing import List, Optional
from pydantic import BaseModel, Field, validator

class CabinetInput(BaseModel):
    cabinet_id: str = Field(..., description="Unique cabinet identifier")
    type: str = Field(..., description="'refrigerator' or 'freezer'")
    air_supply_c: Optional[float] = None
    air_return_c: Optional[float] = None
    prod_t_mw_chill_c: Optional[float] = None
    prod_t_milk_c: Optional[float] = None
    prod_t_mw_freeze_c: Optional[float] = None
    defrost_status: int = 0
    time_since_defrost_min: int = 0

    @validator("type")
    def _valid_type(cls, v):
        assert v in {"refrigerator", "freezer"}
        return v

    @validator("defrost_status")
    def _valid_defrost(cls, v):
        assert v in (0, 1)
        return v

    @validator("time_since_defrost_min")
    def _non_negative(cls, v):
        assert v >= 0
        return v

class EvaluateRequest(BaseModel):
    store_id: str
    timestamp: str
    business_hours_flag: int = 1
    cabinets: List[CabinetInput]
