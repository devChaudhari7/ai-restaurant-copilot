from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from utils.data_loader import POSDataStore
from backend.pos_analysis import (
    compute_menu_engineering_metrics,
    get_menu_insights,
    get_price_optimization_suggestions,
    get_sales_patterns,
)
from backend.recommendation import get_top_combos, get_upsell_suggestions
from backend.voice_order import parse_order_text


app = FastAPI(title="AI-Powered Revenue & Voice Copilot for Restaurants")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OrderItem(BaseModel):
    name: str
    qty: int = 1
    size: Optional[str] = None


class ParsedOrder(BaseModel):
    items: List[OrderItem]


class VoiceOrderRequest(BaseModel):
    text: str


class CreateOrderRequest(BaseModel):
    items: List[OrderItem]


class CreateOrderResponse(BaseModel):
    order_id: str
    items: List[OrderItem]
    upsell_suggestions: List[str] = []


pos_store = POSDataStore()


@app.post("/upload-pos-data")
async def upload_pos_data(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    content = await file.read()
    pos_store.load_from_bytes(content)

    if pos_store.df is None:
        raise HTTPException(status_code=400, detail="Failed to load POS data.")

    analytics = compute_menu_engineering_metrics(pos_store.df)
    sales_patterns = get_sales_patterns(pos_store.df)

    return {
        "status": "ok",
        "rows": len(pos_store.df),
        "analytics_preview": analytics.head(20).to_dict(orient="records"),
        "sales_patterns": sales_patterns,
    }


@app.get("/menu-insights")
async def menu_insights() -> Dict[str, Any]:
    if pos_store.df is None:
        raise HTTPException(status_code=400, detail="POS data not loaded. Upload first.")

    insights = get_menu_insights(pos_store.df)
    return {"status": "ok", "menu_insights": insights}


@app.get("/combo-recommendations")
async def combo_recommendations() -> Dict[str, Any]:
    if pos_store.df is None:
        raise HTTPException(status_code=400, detail="POS data not loaded. Upload first.")

    combos = get_top_combos(pos_store.df)
    return {"status": "ok", "combos": combos}


@app.get("/price-optimization")
async def price_optimization() -> Dict[str, Any]:
    if pos_store.df is None:
        raise HTTPException(status_code=400, detail="POS data not loaded. Upload first.")

    suggestions = get_price_optimization_suggestions(pos_store.df)
    return {"status": "ok", "suggestions": suggestions}


@app.get("/sales-patterns")
async def sales_patterns() -> Dict[str, Any]:
    if pos_store.df is None:
        raise HTTPException(status_code=400, detail="POS data not loaded. Upload first.")

    patterns = get_sales_patterns(pos_store.df)
    return {"status": "ok", "sales_patterns": patterns}


@app.post("/voice-order", response_model=ParsedOrder)
async def voice_order(request: VoiceOrderRequest) -> ParsedOrder:
    if pos_store.df is None:
        raise HTTPException(status_code=400, detail="POS data not loaded. Upload first.")

    parsed = parse_order_text(request.text, pos_store.get_unique_items())
    return ParsedOrder(items=[OrderItem(**item) for item in parsed["items"]])


@app.post("/create-order", response_model=CreateOrderResponse)
async def create_order(request: CreateOrderRequest) -> CreateOrderResponse:
    if pos_store.df is None:
        raise HTTPException(status_code=400, detail="POS data not loaded. Upload first.")

    order_id = pos_store.generate_order_id()
    item_names = [item.name for item in request.items]
    upsell = get_upsell_suggestions(item_names, pos_store.df)

    # Simulate sending order to POS (here we just echo back)
    return CreateOrderResponse(order_id=order_id, items=request.items, upsell_suggestions=upsell)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

