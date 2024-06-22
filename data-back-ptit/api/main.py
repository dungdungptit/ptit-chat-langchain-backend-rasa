from fastapi import FastAPI, Depends
import crud, models, schemas
from database import SessionLocal, engine
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

origins = [
    "*",
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/bins/{id}")
def get_bin_by_id(id: int, db: Session = Depends(get_db)):
    bin = crud.get_bin_by_id(db, id)
    return {
        "message": "success",
        "data": {
            "id": id,
            "updated_at": 1692932662542,
            "weight": bin.weight,
            "unit": bin.unit,
        },
    }


@app.get("/vehicle")
def get_tracking(db: Session = Depends(get_db)):
    vehicle = crud.get_tracking(db)
    xxx = {
        "message": "success",
        "length": "1",
        "data": [
            {
                "timestamp": 1692928842085,
                "location": {"type": "Point", "coordinates": [vehicle.lng, vehicle.lat]},
            }
        ],
    }
    return xxx

@app.post("/bins")
def create_bin(bin: schemas.BinsCreate, db: Session = Depends(get_db)):
    return crud.create_bin(db, bin)

@app.post("/vehicle")
def create_vehicle(vehicle: schemas.VehiclesCreate, db: Session = Depends(get_db)):
    return crud.create_vehicle(db, vehicle)

@app.put("/bins/{id}")
def update_bin(id: int, bin: schemas.BinsCreate, db: Session = Depends(get_db)):
    return crud.update_bin(db, bin, id)

@app.put("/vehicle")
def update_vehicle(vehicle: schemas.VehiclesCreate, db: Session = Depends(get_db)):
    return crud.update_vehicle(db, vehicle)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9998)