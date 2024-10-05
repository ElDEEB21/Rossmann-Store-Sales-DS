from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from model_pipeline import Predict
import pandas as pd

app = FastAPI()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="D:\model\templates")

# Define request model for API
class SalesPredictionInput(BaseModel):
    store: int
    day_of_week: int
    date: str
    customers: int
    open_store: int
    promo: int
    state_holiday: str
    school_holiday: int
    store_type: str
    assortment: str
    competition_distance: float
    competition_open_month: int
    competition_open_year: int
    promo2: int
    promo2_since_week: int
    promo2_since_year: int
    promo_interval: str

# UI Route (Homepage)
@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("D:\model\templates\form.html", {"request": request})

# Form submission (UI POST request)
@app.post("/submit", response_class=HTMLResponse)
async def submit_form(request: Request,
                      store: int = Form(...),
                      day_of_week: int = Form(...),
                      date: str = Form(...),
                      customers: int = Form(...),
                      open_store: int = Form(...),
                      promo: int = Form(...),
                      state_holiday: str = Form(...),
                      school_holiday: int = Form(...),
                      store_type: str = Form(...),
                      assortment: str = Form(...),
                      competition_distance: float = Form(...),
                      competition_open_month: int = Form(...),
                      competition_open_year: int = Form(...),
                      promo2: int = Form(...),
                      promo2_since_week: int = Form(...),
                      promo2_since_year: int = Form(...),
                      promo_interval: str = Form(...)):
    # Convert date to pandas datetime
    date = pd.to_datetime(date)

    # Prepare input data
    values = [
        store, day_of_week, date, customers, open_store, promo, state_holiday, school_holiday,
        store_type, assortment, competition_distance, competition_open_month, competition_open_year,
        promo2, promo2_since_week, promo2_since_year, promo_interval
    ]

    # Instantiate prediction model and process the data
    model_pipeline = Predict(values)
    data = model_pipeline.preporcessing()
    data = model_pipeline.feature_enginnering(data)
    data = model_pipeline.pipline(data)

    # Make prediction
    predicted_sales = model_pipeline.prectict_model(data)

    # Return the result with the form again
    return templates.TemplateResponse("form.html", {
        "request": request,
        "predicted_sales": round(predicted_sales[0], 2)
    })

# Run FastAPI server: uvicorn modelapi --reload
