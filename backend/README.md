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


