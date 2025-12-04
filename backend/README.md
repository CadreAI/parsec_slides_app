# Backend Flask Application

Python Flask backend for data ingestion and chart generation.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set up Google Cloud credentials:
    - Place your service account JSON file in the project root or set `GOOGLE_APPLICATION_CREDENTIALS` environment variable
    - Update `config.gcp.credentials_path` if using a custom path

3. Run the Flask server:

```bash
python app.py
```

The server will start on `http://localhost:5000` by default.

## API Endpoints

### `GET /health`

Health check endpoint.

### `POST /ingest-and-generate`

Combined endpoint for data ingestion and chart generation.

**Request Body:**

```json
{
    "partnerName": "Partner Name",
    "outputDir": "./charts",
    "config": {
        "gcp": {
            "project_id": "your-project-id",
            "location": "US"
        },
        "sources": {
            "nwea": {
                "table_id": "project.dataset.table"
            }
        },
        "paths": {
            "data_dir": "./data",
            "output_dir": "./charts"
        }
    },
    "chartFilters": {
        "districts": ["District 1"],
        "schools": ["School 1"],
        "years": [2023, 2024],
        "grades": [1, 2, 3]
    }
}
```

**Response:**

```json
{
    "success": true,
    "charts": ["path/to/chart1.png", "path/to/chart2.png"],
    "summary": {
        "nwea": {
            "rows": 1000,
            "columns": 50
        }
    }
}
```

## Environment Variables

- `PORT`: Server port (default: 5000)
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to GCP service account JSON file
- `OPENAI_API_KEY`: OpenAI API key for chart analysis (required for ChatGPT features)

### Setting Environment Variables

You can set environment variables in several ways:

#### Option 1: Using a `.env` file (Recommended for Development)

Create a `.env` file in the `backend/` directory or project root:

```bash
# backend/.env
OPENAI_API_KEY=sk-your-api-key-here
PORT=5000
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

The `.env` file is automatically loaded when the Flask app starts (if `python-dotenv` is installed).

#### Option 2: Export in Terminal (For Production)

```bash
export OPENAI_API_KEY="sk-your-api-key-here"
export PORT=5000
python app.py
```

#### Option 3: Set in Shell Profile (Persistent)

Add to your `~/.zshrc` or `~/.bashrc`:

```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

Then reload:

```bash
source ~/.zshrc  # or source ~/.bashrc
```

### Getting Your OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key (it starts with `sk-`)
5. Store it securely - you won't be able to see it again!

**Note:** The `.env` file is already in `.gitignore` to keep your API key secure.
