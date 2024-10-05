from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from model_pipeline import Predict
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory=r"D:\model\templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict/")
async def predict(
    request: Request,
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
    promo_interval: str = Form(...)
):
    # Prepare input data
    values = [
        store, day_of_week,date,customers, open_store, promo, state_holiday, school_holiday,
        store_type, assortment, competition_distance, competition_open_month, competition_open_year,
        promo2, promo2_since_week, promo2_since_year, promo_interval
    ]

    # Instantiate the prediction class and run the model
    model_pipeline = Predict(values)
    data = model_pipeline.preporcessing()
    data = model_pipeline.feature_enginnering(data)
    data = model_pipeline.pipline(data)
    predict_model = model_pipeline.prectict_model(data)

    # Return the template with the predicted sales
    sales_prediction = f"{predict_model[0]:,.2f}"
    return templates.TemplateResponse("form.html", {
        "request": request,
        "sales": sales_prediction
    })
