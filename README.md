# Parsec Slides App

A comprehensive educational data visualization and presentation system that generates Google Slides presentations from assessment data (NWEA, STAR, iReady). The system includes AI-powered chart analysis, intelligent chart selection, and automated slide generation.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Components](#components)
- [Setup Instructions](#setup-instructions)
- [Determination Engine & Training Data](#determination-engine--training-data)
- [Key Workflows](#key-workflows)
- [Environment Variables](#environment-variables)

## Overview

This application provides an end-to-end solution for:

- **Data Ingestion**: Pulling assessment data from BigQuery
- **Chart Generation**: Creating visualizations for NWEA, STAR, and iReady assessments
- **AI Analysis**: Generating insights using GPT-4 with Emergent Learning framework
- **Slide Creation**: Automatically creating Google Slides presentations
- **Intelligent Selection**: Using LLM to determine which charts to include based on user prompts

## Repository Structure

```
parsec_slides_app/
├── backend/                    # Flask backend API
│   ├── app.py                  # Main Flask application
│   ├── celery_app.py           # Celery task queue configuration
│   ├── requirements.txt         # Python dependencies
│   └── python/                 # Core Python modules
│       ├── data_ingestion.py   # BigQuery data ingestion
│       ├── chart_analyzer.py   # AI-powered chart analysis
│       ├── decision_llm.py     # Determination engine (LLM-based decisions)
│       ├── bigquery_client.py  # BigQuery client wrapper
│       ├── google_slides_client.py  # Google Slides API client
│       ├── google_drive_upload.py   # Google Drive upload utilities
│       ├── nwea/               # NWEA chart generation
│       ├── star/               # STAR chart generation
│       ├── iready/             # iReady chart generation
│       ├── slides/              # Slide creation logic
│       ├── reference_decks/    # Reference PDFs for training
│       └── tasks/               # Celery background tasks
├── frontend/                    # Next.js frontend
│   ├── src/
│   │   ├── app/                # Next.js app router pages
│   │   ├── components/         # React components
│   │   ├── hooks/              # Custom React hooks
│   │   ├── lib/                # Utility libraries
│   │   └── types/              # TypeScript type definitions
│   └── package.json            # Node.js dependencies
└── README.md                   # This file
```

## Components

### Backend Components

#### 1. **Flask API (`backend/app.py`)**

Main REST API endpoints:

- `/health` - Health check
- `/config/assessment-filters` - Get available assessment filters
- `/config/student-groups` - Get student group definitions
- `/data/ingest` - Ingest assessment data from BigQuery
- `/charts/generate` - Generate charts for assessments
- `/slides/create` - Create Google Slides presentation

#### 2. **Data Ingestion (`backend/python/data_ingestion.py`)**

- Pulls data from BigQuery for NWEA, STAR, and iReady assessments
- Handles filtering by district, school, grade, subject, quarters (BOY/MOY/EOY)
- **Quarter Mapping**: BOY → Fall, MOY → Winter, EOY → Spring (for data filtering)
- Supports district-only filtering option
- Normalizes column names and data formats
- Supports concurrent queries for performance (up to 5 parallel BigQuery queries)

#### 3. **Chart Generation**

- **NWEA Charts** (`backend/python/nwea/nwea_charts.py`): Generates year-over-year trend charts, student group comparisons, grade-level dashboards
    - Uses BOY/MOY/EOY naming in titles and filenames
    - Always generates individual charts per grade (no consolidated charts for cohort trends)
- **STAR Charts** (`backend/python/star/star_charts.py`): Generates performance progression charts, benchmark achievement charts, growth metrics
    - Uses BOY/MOY/EOY naming in titles and filenames
    - Always generates individual charts per grade (no consolidated charts for cohort trends or SGP growth)
    - Each grade gets its own separate chart file
- **iReady Charts** (`backend/python/iready/iready_charts.py`): Generates diagnostic results, growth tracking, placement charts
    - Uses BOY/MOY/EOY naming in titles and filenames
    - Always generates individual charts per grade

#### 4. **Chart Analyzer (`backend/python/chart_analyzer.py`)**

- Uses GPT-4 (text-only, data-based analysis) - **NO image analysis**
- Analyzes charts using structured JSON data files generated alongside chart images
- Implements Emergent Learning framework:
    - **Ground Truths**: Observable facts with specific numbers
    - **Insights**: Patterns and meanings derived from data
    - **Hypotheses**: Forward-looking predictions
    - **Opportunities**: Actionable recommendations at classroom/grade/school/system levels
- Uses reference decks for context and style guidance (filtered by deck type: BOY/MOY/EOY)
- **Optimized for Performance**:
    - Batches multiple charts per API call (8 charts per call by default)
    - Parallel processing (up to 3 concurrent API calls)
    - Token estimation and data summarization for large datasets
    - Automatic model selection (GPT-4 vs GPT-3.5-turbo) based on prompt size
- Generates structured JSON output with analysis results

#### 5. **Determination Engine (`backend/python/decision_llm.py`)**

Two main functions:

**a) `should_use_ai_insights()`**

- Determines whether to use AI insights based on user prompt
- Considers: chart count, user preferences, cost implications
- Returns: `use_ai`, `reasoning`, `confidence`, `analysis_focus`
- **Note**: Does NOT use reference decks - makes decisions based on prompt only

**b) `parse_chart_instructions()`**

- Parses user prompts to determine which charts to include and their order
- Handles natural language instructions like:
    - "all graphs" → includes all charts
    - "grades 1-4 math and reading" → filters by grade/subject
    - "show Hispanic student trends" → includes demographic charts
- Returns: `chart_selection`, `instructions`, `reasoning`
- **Note**: Uses hardcoded ordering priorities (section3 > section1 > section4 > section2 > section0 > section6) rather than learning from reference decks
- **Future Enhancement**: Could analyze reference deck slide orders to learn preferred layouts

#### 6. **Slide Creator (`backend/python/slides/slide_creator.py`)**

- Creates Google Slides presentations
- **Slide Types**:
    - **Single Chart Slides**: One chart per slide with title and summary
    - **Dual Chart Slides**: Math + Reading pairs on same slide (same grade, same scope)
    - **No Triple Chart Slides**: Removed - only single or dual charts supported
- Handles chart pairing logic (matches math/reading charts by grade and scope)
- Manages slide layout and formatting
- Integrates AI insights into slides (if enabled)
- Uploads charts to Google Drive in batches
- Filters charts based on user prompt and reference deck patterns

#### 7. **Celery Tasks (`backend/python/tasks/slides.py`)**

- Background task processing for slide creation
- Handles long-running operations asynchronously
- Provides task status tracking
- Configurable time limits (default: 30 minutes for slide creation)
- Error handling for timeout scenarios

### Frontend Components

#### 1. **Next.js Application (`frontend/`)**

- **Pages**:
    - `/dashboard` - Main dashboard
    - `/create-deck` - Deck creation interface
    - `/sign-in` - Authentication (Clerk)
- **Features**:
    - Assessment filter selection (NWEA, STAR, iReady)
    - Quarter selection (BOY, MOY, EOY)
    - District-only filtering toggle
    - Grade, subject, student group, and race/ethnicity selection
    - User prompt input for chart selection and AI insights

#### 2. **API Routes (`frontend/src/app/api/`)**

- `/bigquery/*` - BigQuery data fetching
- `/data/ingest` - Data ingestion trigger
- `/slides/create` - Slide creation trigger
- `/tasks/*` - Task status tracking

#### 3. **Hooks (`frontend/src/hooks/`)**

- `useAssessmentFilters.ts` - Assessment filter management
- `useDistrictsAndSchools.ts` - District/school selection
- `useStudentGroups.ts` - Student group selection
- `useFormOptions.ts` - Form option management

#### 4. **Components (`frontend/src/components/ui/`)**

- Reusable UI components (buttons, cards, selects, etc.)
- Built with Radix UI and Tailwind CSS

## Setup Instructions

### Prerequisites

- Python 3.12+ (or 3.13+)
- Node.js 18+ (or Bun)
- Google Cloud Platform account with BigQuery access
- Google Cloud credentials (service account JSON)
- OpenAI API key
- Supabase account (for database)
- Redis (for Celery task queue)

### Backend Setup

1. **Create virtual environment:**

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
   Create `backend/.env` or `.env` in project root:

```env
# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=path/to/service_account.json
GOOGLE_CLOUD_PROJECT=your-project-id

# OpenAI
OPENAI_API_KEY=your-openai-api-key

# Supabase
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-key

# Redis (for Celery)
REDIS_URL=redis://localhost:6379/0

# Google Drive/Slides
DEFAULT_SLIDES_FOLDER_ID=your-google-drive-folder-id
```

4. **Run Flask development server:**

```bash
cd backend
python app.py
# Or with gunicorn:
gunicorn app:app --bind 0.0.0.0:5000
```

5. **Run Celery worker (for background tasks):**

```bash
cd backend
celery -A celery_app worker --loglevel=info
```

### Frontend Setup

1. **Install dependencies:**

```bash
cd frontend
bun install  # or npm install
```

2. **Set up environment variables:**
   Create `frontend/.env.local`:

```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:5000
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your-clerk-key
CLERK_SECRET_KEY=your-clerk-secret
NEXT_PUBLIC_SUPABASE_URL=your-supabase-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
```

3. **Run development server:**

```bash
cd frontend
bun dev  # or npm run dev
```

## Determination Engine & Training Data

### Overview

The **Determination Engine** (`decision_llm.py`) uses GPT-3.5-turbo to make intelligent decisions about:

1. Whether to use AI insights (based on user preferences and chart count)
2. Which charts to include in presentations (based on natural language instructions)

### Deck Type Organization

Reference decks are organized by deck type in separate folders:

- `backend/python/reference_decks/BOY-DECKS/` - Beginning of Year reference decks
- `backend/python/reference_decks/MOY-DECKS/` - Middle of Year reference decks
- `backend/python/reference_decks/EOY-DECKS/` - End of Year reference decks

The system automatically selects the appropriate reference deck folder based on the selected quarters (BOY, MOY, or EOY) when creating a presentation. This ensures that:

- BOY decks train on BOY reference patterns
- MOY decks train on MOY reference patterns
- EOY decks train on EOY reference patterns

### Normalization Requirements

To improve the determination engine's accuracy, you need to:

#### 1. **Normalize Chart Filenames**

Chart filenames should follow a consistent pattern for the LLM to parse:

```
{scope}_{section}_{details}_{subject}_{window}_{type}.png
```

Examples:

- `district_section1_boy_trends.png` (or `fall_trends.png` for internal data filtering)
- `school_section3_grade_1_math_boy_trends.png`
- `district_section2_hispanic_reading_boy_trends.png`

**Required components:**

- `scope`: `district` or `school`
- `section`: `section0`, `section1`, `section2`, `section3`, etc.
- `subject`: `math`, `reading`, `math_reading` (for dual charts)
- `window`: `boy`, `moy`, `eoy` (display names) - maps to `fall`, `winter`, `spring` for data filtering
- `type`: `trends`, `cohort`, `sgp`, `growth`, etc.
- `grade`: Individual grade numbers (e.g., `grade_1`, `grade_2`) - **no consolidated grade ranges**

#### 2. **Normalize User Prompts**

The system should handle common variations:

- **"All charts"**: `all graphs`, `all charts`, `all of them`, `everything`, `include all`, `show all`, `output all`
- **Grade ranges**: `grades 1-4`, `grade 1-4`, `grades 1 through 4`
- **Subjects**: `math`, `mathematics`, `reading`, `ela`
- **Demographics**: `Hispanic`, `Latino`, `Black`, `African American`, `White`, `demographic`, `student group`
- **Sections**: `section1`, `section 1`, `trends`, `year to year`

#### 3. **Reference Deck Organization**

Reference decks are organized by deck type in separate folders:

**BOY-DECKS/** (Beginning of Year):

- `ADAMSAMPLE_Bridges BOY 2025 UPDATE.pdf`
- `PATRICKSAMPLE_2026_Insight Deck_BOY_Plumas Charter.pdf`
- `PATRICKSAMPLE_2026_Insights Deck_BOY_Big Picture.pdf`
- `2026_Insight Deck_BOY_YPICS.pdf`
- `Bridges Charter 2024_Insight Deck_Q2_Fall NWEA.pdf`

**MOY-DECKS/** (Middle of Year):

- `2025_Insight Deck_MOY_Alta Public Schools.pdf`
- `2025-26 MOY Insight Deck LPS.pdf`
- `2025-26 MOY Insight Deck Strathmore.pdf`
- `2026_Insight Deck_MOY_Monterey Bay Charter School.pdf`

**EOY-DECKS/** (End of Year):

- `PATRICKSAMPLE_2025_Insight Deck_EOY_Chowchilla Elementary.pdf`
- `2025_Insight Deck_EOY_San Ardo Union.pdf`
- `2025_Insight Deck_EOY_Twin Ridges.pdf`
- `2025_Insight Deck_EOY_Yosemite USD.pdf`
- `Santa Rita EOY 2024-25.pdf`

**Naming Convention:**

- Follow consistent naming: `{PARTNER}_{ASSESSMENT}_{QUARTER}_{YEAR}_{SCOPE}.pdf`
- Contain example insights following Emergent Learning framework
- Include structured analysis examples (ground truths, insights, hypotheses, opportunities)

**Deck Type Selection:**

- When creating a deck, the system automatically detects deck type from selected quarters
- BOY quarters → uses BOY-DECKS folder
- MOY quarters → uses MOY-DECKS folder
- EOY quarters → uses EOY-DECKS folder
- If multiple quarters selected, prioritizes: EOY > MOY > BOY

### Training Data Requirements

To improve the determination engine, collect and structure:

#### 1. **Decision Training Examples**

Create a dataset of user prompts → decisions:

```json
{
  "user_prompt": "Show me all math charts for grades 1-4",
  "chart_count": 50,
  "expected_decision": {
    "use_ai": false,
    "reasoning": "User wants specific subset, no analysis needed",
    "chart_selection": ["section3_grade1_math_*.png", "section3_grade2_math_*.png", ...]
  },
  "actual_decision": {...}
}
```

#### 2. **Chart Selection Training Examples**

Examples of natural language → chart selection:

```json
{
  "user_prompt": "I want to see Hispanic student performance trends",
  "available_charts": ["district_section2_hispanic_math_fall_trends.png", ...],
  "expected_selection": ["district_section2_hispanic_*.png", "school_section2_hispanic_*.png"],
  "expected_order": ["district", "school"]
}
```

#### 3. **Insight Quality Examples**

Examples of good vs. bad insights:

```json
{
    "chart_type": "section3_grade1_math_fall_trends",
    "good_insight": {
        "finding": "Math scores increased 8% from 2022 to 2023",
        "implication": "Instructional changes are showing positive impact",
        "recommendation": "Continue current math curriculum approach"
    },
    "bad_insight": {
        "finding": "There are some numbers on the chart",
        "implication": "This is data",
        "recommendation": "Look at the chart"
    }
}
```

### Improving the Determination Engine

1. **Collect Training Data:**
    - Log all user prompts and decisions
    - Collect feedback on chart selections
    - Track insight quality ratings

2. **Fine-tune Prompts:**
    - Update `decision_prompt` in `should_use_ai_insights()` based on common failure modes
    - Update `selection_prompt` in `parse_chart_instructions()` to handle edge cases

3. **Add Reference Examples:**
    - Add more reference decks with diverse analysis styles
    - Include examples of different assessment types (NWEA, STAR, iReady)
    - Cover different quarters (BOY, MOY, EOY)

4. **Implement Feedback Loop:**
    - Store user corrections to chart selections
    - Track which insights were most useful
    - Use feedback to improve prompt engineering

5. **Consider Fine-tuning:**
    - If you have enough training data (1000+ examples), consider fine-tuning GPT-3.5-turbo
    - Create a fine-tuning dataset with prompt-completion pairs
    - Deploy fine-tuned model for better accuracy

6. **Layout Learning from Reference Decks (IMPLEMENTED):**
    - ✅ **New Feature**: Reference decks are now analyzed to learn layout patterns
    - ✅ **Deck Type Filtering**: System automatically uses BOY-DECKS, MOY-DECKS, or EOY-DECKS based on selected quarters
    - ✅ Extracts slide order from reference PDFs using PDF text parsing (PyPDF2, pdfplumber, pymupdf)
    - ✅ Learns section ordering, scope preferences, and subject pairing patterns
    - ✅ Learns chart selection patterns (required vs optional charts)
    - ✅ Learns chart groupings and presentation flow
    - ✅ Updates `parse_chart_instructions()` to use learned layouts as context
    - ✅ Filters charts based on reference deck patterns (omits charts not in reference decks)
    - ✅ Falls back to hardcoded ordering if no reference decks available
    - **Usage**: Run `python backend/python/test_layout_learner.py` to test layout extraction
    - **Deck Type Detection**: Automatically detects from `quarters` parameter (BOY/MOY/EOY)
    - **Chart Analysis**: Uses reference decks to provide context for AI chart analysis (Emergent Learning framework)

### Current Limitations

- **No fine-tuning**: Currently uses zero-shot GPT-3.5-turbo/GPT-4
- **Limited reference decks**: Only reference PDFs in BOY-DECKS, MOY-DECKS, EOY-DECKS folders
- **No feedback mechanism**: No way to learn from user corrections
- **Prompt-based only**: No structured training data pipeline
- **Layout learning**: Reference decks ARE used for chart selection and ordering, but extraction accuracy depends on PDF quality
- **No image analysis**: Chart analysis uses structured JSON data only (no GPT-4o Vision)
- **Single worker optimization**: Optimized for single-worker environments (3 concurrent API calls, 8 charts per batch)

## Key Workflows

### 1. Data Ingestion Workflow

```
User selects filters → Frontend calls /data/ingest → Backend queries BigQuery →
Data normalized → Stored in memory/temp files → Charts generated
```

### 2. Chart Generation Workflow

```
Assessment data → Chart generation module (nwea/star/iready) →
Matplotlib charts created → Saved as PNG → Chart metadata saved as JSON
```

### 3. Slide Creation Workflow

```
User selects filters + provides prompt → Data ingested from BigQuery →
Charts generated (individual per grade) → Decision LLM determines AI usage →
Chart selection LLM filters charts based on prompt + reference deck patterns →
Charts uploaded to Drive in batches → Slides created (single or dual charts only) →
AI insights added (if enabled) → Presentation complete
```

### 4. AI Analysis Workflow

```
Chart + JSON data file → Chart analyzer → Load reference decks (filtered by deck type) →
Build Emergent Learning prompt → Batch analysis (8 charts per API call) →
Parallel processing (3 concurrent calls) → GPT-4/GPT-3.5-turbo analysis →
Structured JSON insights → Added to slides
```

## Environment Variables

### Backend

| Variable                         | Description                       | Required |
| -------------------------------- | --------------------------------- | -------- |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account JSON  | Yes      |
| `GOOGLE_CLOUD_PROJECT`           | GCP project ID                    | Yes      |
| `OPENAI_API_KEY`                 | OpenAI API key                    | Yes      |
| `SUPABASE_URL`                   | Supabase project URL              | Yes      |
| `SUPABASE_KEY`                   | Supabase service role key         | Yes      |
| `REDIS_URL`                      | Redis connection URL              | Yes      |
| `DEFAULT_SLIDES_FOLDER_ID`       | Google Drive folder ID for slides | Yes      |

### Frontend

| Variable                            | Description                | Required |
| ----------------------------------- | -------------------------- | -------- |
| `NEXT_PUBLIC_BACKEND_URL`           | Backend API URL            | Yes      |
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | Clerk auth publishable key | Yes      |
| `CLERK_SECRET_KEY`                  | Clerk auth secret key      | Yes      |
| `NEXT_PUBLIC_SUPABASE_URL`          | Supabase project URL       | Yes      |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY`     | Supabase anonymous key     | Yes      |

## Recent Changes

### Quarter System Update (BOY/MOY/EOY)

- Changed from Fall/Winter/Spring to BOY/MOY/EOY for user-facing selections
- Internal data filtering still uses Fall/Winter/Spring mapping
- All chart titles and filenames use BOY/MOY/EOY naming

### Chart Generation Updates

- **Removed consolidated charts**: All cohort trends and SGP growth charts are now individual per grade
- **No triple chart slides**: Removed support for 3-chart slides - only single and dual charts supported
- **Individual grade charts**: Each grade gets its own separate chart file (no grouping)

### Performance Optimizations

- **Batched chart analysis**: 8 charts analyzed per API call (reduces API calls by ~87%)
- **Parallel processing**: Up to 3 concurrent API calls for faster analysis
- **Optimized for single-worker**: Designed for Render.com single-worker environments
- **Token management**: Automatic data summarization and model selection based on prompt size

### Analysis Improvements

- **Data-only analysis**: Uses structured JSON data files instead of image analysis
- **Reference deck filtering**: Automatically selects appropriate reference decks based on deck type (BOY/MOY/EOY)
- **Layout learning**: Extracts chart ordering and selection patterns from reference PDFs

## Contributing

When adding new features:

1. **Chart Types**: Add generation logic in respective `{assessment}/` folder
2. **AI Analysis**: Update `chart_analyzer.py` prompts
3. **Determination Engine**: Update `decision_llm.py` prompts and add training examples
4. **Reference Decks**: Add new PDFs to appropriate `reference_decks/{BOY|MOY|EOY}-DECKS/` folder following naming convention
5. **Chart Generation**: Always generate individual charts per grade (no consolidated charts)
6. **Slide Creation**: Only create single or dual chart slides (no triple charts)

## License

[Add your license here]
