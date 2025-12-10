"""
Celery tasks for background processing
"""
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Set matplotlib backend before any imports
import matplotlib
matplotlib.use('Agg')

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent / 'python'))

from celery_app import celery_app
from python.data_ingestion import ingest_nwea, ingest_iready, ingest_star
from python.nwea.nwea_charts import generate_nwea_charts
from python.iready.iready_charts import generate_iready_charts
from python.star.star_charts import generate_star_charts
from python.slides import create_slides_presentation
from python.supabase_client import get_supabase_client


@celery_app.task(bind=True, name='tasks.ingest_and_generate_charts')
def ingest_and_generate_charts_task(self, partner_name, config, chart_filters, output_dir='./data'):
    """
    Background task for data ingestion and chart generation
    
    Args:
        partner_name: Partner name
        config: Partner configuration dict
        chart_filters: Chart filters dict
        output_dir: Output directory path
    
    Returns:
        Dict with success, charts, summary, and task_id
    """
    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={'stage': 'starting', 'progress': 0})
        
        # Check which data sources are configured
        sources = config.get('sources', {})
        has_nwea = bool(sources.get('nwea'))
        has_iready = bool(sources.get('iready'))
        has_star = bool(sources.get('star'))
        
        if not has_nwea and not has_iready and not has_star:
            raise ValueError('At least one data source (nwea, iready, or star) must be configured')
        
        self.update_state(state='PROGRESS', meta={'stage': 'ingesting_data', 'progress': 10})
        
        # Ingest data for configured sources
        nwea_data = []
        iready_data = []
        star_data = []
        
        if has_nwea:
            self.update_state(state='PROGRESS', meta={'stage': 'ingesting_nwea', 'progress': 20})
            nwea_data = ingest_nwea(
                partner_name=partner_name,
                config=config,
                chart_filters=chart_filters
            )
        
        if has_iready:
            self.update_state(state='PROGRESS', meta={'stage': 'ingesting_iready', 'progress': 40})
            iready_data = ingest_iready(
                partner_name=partner_name,
                config=config,
                chart_filters=chart_filters
            )
        
        if has_star:
            self.update_state(state='PROGRESS', meta={'stage': 'ingesting_star', 'progress': 50})
            star_data = ingest_star(
                partner_name=partner_name,
                config=config,
                chart_filters=chart_filters
            )
        
        # Generate charts
        self.update_state(state='PROGRESS', meta={'stage': 'generating_charts', 'progress': 60})
        data_dir = config.get('paths', {}).get('data_dir', './data')
        
        # Create temporary directory for charts
        temp_charts_dir = tempfile.mkdtemp(prefix='parsec_charts_')
        
        try:
            all_chart_paths = []
            
            if nwea_data:
                self.update_state(state='PROGRESS', meta={'stage': 'generating_nwea', 'progress': 70})
                nwea_charts = generate_nwea_charts(
                    partner_name=partner_name,
                    output_dir=temp_charts_dir,
                    config=config,
                    chart_filters=chart_filters,
                    data_dir=data_dir,
                    nwea_data=nwea_data
                )
                all_chart_paths.extend(nwea_charts)
            
            if iready_data:
                self.update_state(state='PROGRESS', meta={'stage': 'generating_iready', 'progress': 80})
                iready_charts = generate_iready_charts(
                    partner_name=partner_name,
                    output_dir=temp_charts_dir,
                    config=config,
                    chart_filters=chart_filters,
                    data_dir=data_dir,
                    iready_data=iready_data
                )
                all_chart_paths.extend(iready_charts)
            
            if star_data:
                self.update_state(state='PROGRESS', meta={'stage': 'generating_star', 'progress': 85})
                star_charts = generate_star_charts(
                    partner_name=partner_name,
                    output_dir=temp_charts_dir,
                    config=config,
                    chart_filters=chart_filters,
                    data_dir=data_dir,
                    star_data=star_data
                )
                all_chart_paths.extend(star_charts)
            
            self.update_state(state='PROGRESS', meta={'stage': 'completed', 'progress': 100})
            
            # Build summary
            summary = {}
            if nwea_data:
                summary['nwea'] = {
                    'rows': len(nwea_data),
                    'columns': len(nwea_data[0]) if nwea_data else 0
                }
            if iready_data:
                summary['iready'] = {
                    'rows': len(iready_data),
                    'columns': len(iready_data[0]) if iready_data else 0
                }
            if star_data:
                summary['star'] = {
                    'rows': len(star_data),
                    'columns': len(star_data[0]) if star_data else 0
                }
            
            return {
                'success': True,
                'charts': all_chart_paths,
                'summary': summary,
                'charts_generated': len(all_chart_paths),
                'temp_dir': temp_charts_dir  # Keep temp dir for now, will be cleaned up later
            }
        except Exception as chart_error:
            # Clean up temp directory on error
            if os.path.exists(temp_charts_dir):
                shutil.rmtree(temp_charts_dir, ignore_errors=True)
            raise chart_error
    except Exception as e:
        error_msg = str(e)
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': error_msg
        }


@celery_app.task(bind=True, name='tasks.create_slides_presentation')
def create_slides_presentation_task(
    self, title, chart_paths, drive_folder_url=None, 
    enable_ai_insights=True, user_prompt=None, clerk_user_id=None, description=None
):
    """
    Background task for creating Google Slides presentation
    
    Args:
        title: Presentation title
        chart_paths: List of chart file paths
        drive_folder_url: Optional Google Drive folder URL
        enable_ai_insights: Whether to use AI insights
        user_prompt: Optional user prompt
        clerk_user_id: Optional Clerk user ID for Supabase
        description: Optional deck description
    
    Returns:
        Dict with presentationId, presentationUrl, title, deckId
    """
    try:
        self.update_state(state='PROGRESS', meta={'stage': 'creating_presentation', 'progress': 0})
        
        # Create the presentation
        result = create_slides_presentation(
            title=title,
            chart_paths=chart_paths,
            drive_folder_url=drive_folder_url,
            enable_ai_insights=enable_ai_insights,
            user_prompt=user_prompt
        )
        
        self.update_state(state='PROGRESS', meta={'stage': 'saving_deck', 'progress': 90})
        
        # Save deck to Supabase (if clerk_user_id provided)
        if clerk_user_id:
            try:
                supabase = get_supabase_client()
                
                deck_data = {
                    'clerk_user_id': clerk_user_id,
                    'title': title,
                    'description': description,
                    'slide_count': result.get('slideCount'),
                    'presentation_id': result.get('presentationId'),
                    'presentation_url': result.get('presentationUrl')
                }
                
                deck_response = supabase.table('decks').insert(deck_data).execute()
                
                if deck_response.data:
                    result['deckId'] = deck_response.data[0]['id']
            except Exception as e:
                print(f"[Celery] Error saving deck to Supabase: {e}")
                # Don't fail the task if Supabase save fails
        
        self.update_state(state='PROGRESS', meta={'stage': 'completed', 'progress': 100})
        
        return result
    except Exception as e:
        error_msg = str(e)
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': error_msg
        }
