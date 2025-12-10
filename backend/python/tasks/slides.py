import logging
import time
import json
import tempfile
import shutil
import os

from celery_app import celery_app


@celery_app.task(bind=True, name="create_deck_with_slides_task")
def create_deck_with_slides_task(
    self,
    partner_name: str,
    config: dict,
    chart_filters: dict,
    title: str,
    clerk_user_id: str,
    drive_folder_url: str = None,
    enable_ai_insights: bool = True,
    user_prompt: str = None,
    description: str = None
):
    """
    Combined task that ingests data, generates charts, and creates a Google Slides presentation.
    Updates task status in Supabase throughout execution.
    """
    from python.supabase_client import get_supabase_client
    from python.data_ingestion import ingest_nwea, ingest_iready, ingest_star
    from python.nwea.nwea_charts import generate_nwea_charts
    from python.iready.iready_charts import generate_iready_charts
    from python.star.star_charts import generate_star_charts
    from python.slides import create_slides_presentation

    task_id = self.request.id
    temp_charts_dir = None

    try:
        logging.info(f"[Task {task_id}] Starting create_deck_with_slides_task for {partner_name}")

        # Update status to STARTED in Supabase
        try:
            supabase = get_supabase_client()
            supabase.table('tasks').update({
                'status': 'STARTED'
            }).eq('celery_task_id', task_id).execute()
            logging.info(f"[Task {task_id}] Updated status to STARTED in DB")
        except Exception as e:
            logging.error(f"[Task {task_id}] Failed to update STARTED status: {e}")

        # Validate inputs
        if not partner_name:
            raise ValueError('partnerName is required')
        if not title:
            raise ValueError('title is required')

        # Handle case where config might be a JSON string
        if isinstance(config, str):
            logging.info(f"[Task {task_id}] Parsing config from JSON string...")
            config = json.loads(config)

        # Ensure config is a dict
        if not isinstance(config, dict):
            raise ValueError(f'config must be a dict, got {type(config).__name__}')

        # Handle case where chart_filters might be a JSON string
        if isinstance(chart_filters, str):
            logging.info(f"[Task {task_id}] Parsing chart_filters from JSON string...")
            try:
                chart_filters = json.loads(chart_filters)
            except json.JSONDecodeError:
                logging.warning(f"[Task {task_id}] Failed to parse chart_filters, using empty dict")
                chart_filters = {}

        # Ensure chart_filters is a dict
        if not isinstance(chart_filters, dict):
            logging.warning(f"[Task {task_id}] chart_filters is not a dict, using empty dict")
            chart_filters = {}

        # Check which data sources are configured
        sources = config.get('sources', {})
        has_nwea = bool(sources.get('nwea'))
        has_iready = bool(sources.get('iready'))
        has_star = bool(sources.get('star'))

        if not has_nwea and not has_iready and not has_star:
            raise ValueError('At least one data source (nwea, iready, or star) must be configured in config.sources')

        logging.info(f"[Task {task_id}] Data sources configured: NWEA={has_nwea}, iReady={has_iready}, STAR={has_star}")

        # ====================
        # STEP 1: Ingest Data
        # ====================
        nwea_data = []
        iready_data = []
        star_data = []

        if has_nwea:
            try:
                nwea_data = ingest_nwea(
                    partner_name=partner_name,
                    config=config,
                    chart_filters=chart_filters
                )
                logging.info(f"[Task {task_id}] NWEA data ingested: {len(nwea_data)} rows")
            except Exception as e:
                logging.error(f"[Task {task_id}] Error ingesting NWEA data: {e}")
                if not has_iready and not has_star:
                    raise
                logging.info(f"[Task {task_id}] Continuing with other sources...")

        if has_iready:
            try:
                iready_data = ingest_iready(
                    partner_name=partner_name,
                    config=config,
                    chart_filters=chart_filters
                )
                logging.info(f"[Task {task_id}] iReady data ingested: {len(iready_data)} rows")
            except Exception as e:
                logging.error(f"[Task {task_id}] Error ingesting iReady data: {e}")
                if not has_nwea and not has_star:
                    raise
                logging.info(f"[Task {task_id}] Continuing with other sources...")

        if has_star:
            try:
                star_data = ingest_star(
                    partner_name=partner_name,
                    config=config,
                    chart_filters=chart_filters
                )
                logging.info(f"[Task {task_id}] STAR data ingested: {len(star_data)} rows")
            except Exception as e:
                logging.error(f"[Task {task_id}] Error ingesting STAR data: {e}")
                if not has_nwea and not has_iready:
                    raise
                logging.info(f"[Task {task_id}] Continuing with other sources...")

        # ====================
        # STEP 2: Generate Charts
        # ====================
        logging.info(f"[Task {task_id}] Starting chart generation...")
        data_dir = config.get('paths', {}).get('data_dir', './data')

        # Create temporary directory for charts
        temp_charts_dir = tempfile.mkdtemp(prefix='parsec_charts_')
        logging.info(f"[Task {task_id}] Created temporary charts directory: {temp_charts_dir}")

        all_chart_paths = []

        # Generate NWEA charts if data is available
        if nwea_data:
            logging.info(f"[Task {task_id}] Generating NWEA charts...")
            nwea_charts = generate_nwea_charts(
                partner_name=partner_name,
                output_dir=temp_charts_dir,
                config=config,
                chart_filters=chart_filters,
                data_dir=data_dir,
                nwea_data=nwea_data
            )
            all_chart_paths.extend(nwea_charts)
            logging.info(f"[Task {task_id}] Generated {len(nwea_charts)} NWEA charts")

        # Generate iReady charts if data is available
        if iready_data:
            logging.info(f"[Task {task_id}] Generating iReady charts...")
            iready_charts = generate_iready_charts(
                partner_name=partner_name,
                output_dir=temp_charts_dir,
                config=config,
                chart_filters=chart_filters,
                data_dir=data_dir,
                iready_data=iready_data
            )
            all_chart_paths.extend(iready_charts)
            logging.info(f"[Task {task_id}] Generated {len(iready_charts)} iReady charts")

        # Generate STAR charts if data is available
        if star_data:
            logging.info(f"[Task {task_id}] Generating STAR charts...")
            star_charts = generate_star_charts(
                partner_name=partner_name,
                output_dir=temp_charts_dir,
                config=config,
                chart_filters=chart_filters,
                data_dir=data_dir,
                star_data=star_data
            )
            all_chart_paths.extend(star_charts)
            logging.info(f"[Task {task_id}] Generated {len(star_charts)} STAR charts")

        logging.info(f"[Task {task_id}] Generated {len(all_chart_paths)} total charts")

        if not all_chart_paths:
            raise ValueError("No charts were generated. Cannot create slides without charts.")

        # ====================
        # STEP 3: Create Slides
        # ====================
        logging.info(f"[Task {task_id}] Creating slides presentation: {title}")
        slides_result = create_slides_presentation(
            title=title,
            chart_paths=all_chart_paths,
            drive_folder_url=drive_folder_url,
            enable_ai_insights=enable_ai_insights,
            user_prompt=user_prompt
        )

        # ====================
        # STEP 4: Save Deck to Supabase
        # ====================
        deck_id = None
        if clerk_user_id:
            try:
                supabase = get_supabase_client()

                deck_data = {
                    'clerk_user_id': clerk_user_id,
                    'title': title,
                    'description': description,
                    'slide_count': slides_result.get('slideCount'),
                    'presentation_id': slides_result.get('presentationId'),
                    'presentation_url': slides_result.get('presentationUrl')
                }

                deck_response = supabase.table('decks').insert(deck_data).execute()

                if deck_response.data:
                    deck_id = deck_response.data[0]['id']
                    logging.info(f"[Task {task_id}] Deck saved to Supabase: {deck_id}")
                else:
                    logging.warning(f"[Task {task_id}] Failed to save deck to Supabase")

            except Exception as e:
                logging.error(f"[Task {task_id}] Error saving deck to Supabase: {e}")
                # Don't fail the task if Supabase save fails

        # Build final result
        result = {
            'success': True,
            'presentationId': slides_result.get('presentationId'),
            'presentationUrl': slides_result.get('presentationUrl'),
            'slideCount': slides_result.get('slideCount'),
            'title': title,
            'deckId': deck_id
        }

        # Update status to SUCCESS in Supabase
        try:
            supabase = get_supabase_client()
            supabase.table('tasks').update({
                'status': 'SUCCESS',
                'result': result
            }).eq('celery_task_id', task_id).execute()
            logging.info(f"[Task {task_id}] Updated status to SUCCESS in DB")
        except Exception as e:
            logging.error(f"[Task {task_id}] Failed to update SUCCESS status: {e}")

        logging.info(f"[Task {task_id}] Task completed successfully")
        return result

    except Exception as e:
        error_message = str(e)
        logging.error(f"[Task {task_id}] Task failed: {error_message}")
        import traceback
        traceback.print_exc()

        # Update status to FAILURE in Supabase
        try:
            supabase = get_supabase_client()
            supabase.table('tasks').update({
                'status': 'FAILURE',
                'error_message': error_message
            }).eq('celery_task_id', task_id).execute()
            logging.info(f"[Task {task_id}] Updated status to FAILURE in DB")
        except Exception as db_err:
            logging.error(f"[Task {task_id}] Failed to update FAILURE status: {db_err}")

        raise

    finally:
        # Clean up temp directory
        if temp_charts_dir and os.path.exists(temp_charts_dir):
            try:
                logging.info(f"[Task {task_id}] Cleaning up temp directory: {temp_charts_dir}")
                shutil.rmtree(temp_charts_dir, ignore_errors=True)
            except Exception as cleanup_err:
                logging.error(f"[Task {task_id}] Failed to cleanup temp directory: {cleanup_err}")
