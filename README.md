## AI-Powered Revenue & Voice Copilot for Restaurants

This project is a hackathon-ready prototype that combines POS analytics, menu engineering, market basket analysis, and an AI voice ordering copilot for restaurants.

### Tech Stack

- **Backend**: FastAPI (Python)
- **AI**: OpenAI GPT (for order parsing), Whisper (for speech-to-text)
- **Data Processing**: Pandas, NumPy, scikit-learn, mlxtend (Apriori)
- **Frontend**: Streamlit dashboard

### Project Structure

- `backend/` – FastAPI app, POS analytics, recommendations, voice ordering
- `ai_modules/` – OpenAI GPT and Whisper helper clients
- `dashboard/` – Streamlit dashboard app
- `data/` – Sample POS dataset and generator
- `utils/` – Shared utilities (POS data loader)

### Setup

1. **Create virtual environment (recommended)**

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set environment variables**

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_api_key_here
BACKEND_URL=http://localhost:8000
```

4. **Generate sample POS data (optional)**

```bash
python data/generate_sample_pos_data.py
```

This creates `data/sample_pos_data.csv` with ~800 synthetic POS rows.

### Running the Backend

From the project root:

```bash
uvicorn backend.main:app --reload
```

FastAPI will be available at `http://localhost:8000` and the interactive docs at `http://localhost:8000/docs`.

### Running the Dashboard

In a separate terminal from the project root:

```bash
streamlit run dashboard/app.py
```

The dashboard provides:

- **Menu Insights page**
  - Upload POS CSV or use a sample dataset
  - View menu performance table (revenue, cost, contribution margin)
  - See menu category (Star / Puzzle / Plowhorse / Dog)
  - Visualize item popularity and contribution margin
  - Display combo recommendations from market basket analysis

- **Voice Ordering page**
  - Upload audio for transcription (Whisper) or type order text
  - Parse orders with GPT into structured JSON
  - Fuzzy-match spoken items to actual menu items
  - Show upsell suggestions based on combo recommendations
  - Display final (simulated) POS order summary

### Key Endpoints (FastAPI)

- `POST /upload-pos-data`  
  Accepts a CSV file, loads POS data, computes menu metrics, and returns analytics preview plus sales patterns.

- `GET /menu-insights`  
  Returns per-item menu engineering metrics and categories.

- `GET /combo-recommendations`  
  Returns top combo suggestions from Apriori-based market basket analysis.

- `POST /voice-order`  
  Accepts text (from Whisper or typed) and returns a structured parsed order with fuzzy-matched item names.

- `POST /create-order`  
  Accepts structured order JSON, simulates POS order creation, and returns an order summary with upsell suggestions.

### Notes

- The project is intentionally modular and hackathon-ready: you can easily plug in a real POS integration, refine pricing rules, or enhance the voice UX.
- Whisper / GPT model names may evolve; adjust `ai_modules/whisper_client.py` and `ai_modules/gpt_client.py` to match your OpenAI account and latest docs.

