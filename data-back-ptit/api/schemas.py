from typing import List, Union

from pydantic import BaseModel

class BinsBase(BaseModel):
    weight: float

class BinsCreate(BinsBase):
    pass

class Bins(BinsBase):
    class Config:
        from_attributes = True

class VehiclesBase(BaseModel):
    lat: float
    lng: float

class VehiclesCreate(VehiclesBase):
    pass

class Vehicles(VehiclesBase):
    class Config:
        from_attributes = True