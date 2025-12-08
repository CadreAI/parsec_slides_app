"""
Celery tasks for async data processing
"""
import json
import tempfile
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, List
from celery import current_task

# Add backend directory to path to import celery_app
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
from celery_app import celery_app

# Import processing functions - paths should be set up by celery_app.py
from data_ingestion import ingest_nwea
from nwea.nwea_charts import generate_nwea_charts


@celery_app.task(bind=True, name='python.tasks.ingest_and_generate_charts')
def ingest_and_generate_charts_task(
    self,
    partner_name: str,
    output_dir: str,
    config: Dict[str, Any],
    chart_filters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Async task for data ingestion and chart generation
    
    Args:
        partner_name: Partner name
        output_dir: Output directory for charts
        config: Partner configuration dict
        chart_filters: Chart filters dict
        
    Returns:
        Dict with success status, charts, and summary
    """
    try:
        # Update task state to STARTED
        self.update_state(
            state='PROGRESS',
            meta={
                'step': 'Starting data ingestion...',
                'progress': 0,
                'status': 'processing'
            }
        )
        
        # Step 1: Ingest data
        self.update_state(
            state='PROGRESS',
            meta={
                'step': 'Ingesting data from BigQuery...',
                'progress': 10,
                'status': 'processing'
            }
        )
        
        nwea_data = ingest_nwea(
            partner_name=partner_name,
            config=config,
            chart_filters=chart_filters
        )
        
        self.update_state(
            state='PROGRESS',
            meta={
                'step': f'Data ingested: {len(nwea_data):,} rows',
                'progress': 30,
                'status': 'processing',
                'rows_ingested': len(nwea_data)
            }
        )
        
        # Step 2: Generate charts
        self.update_state(
            state='PROGRESS',
            meta={
                'step': 'Generating charts...',
                'progress': 40,
                'status': 'processing'
            }
        )
        
        data_dir = config.get('paths', {}).get('data_dir', './data')
        
        # Create temporary directory for charts
        temp_charts_dir = tempfile.mkdtemp(prefix='parsec_charts_')
        
        try:
            chart_paths = generate_nwea_charts(
                partner_name=partner_name,
                output_dir=temp_charts_dir,
                config=config,
                chart_filters=chart_filters,
                data_dir=data_dir,
                nwea_data=nwea_data
            )
            
            self.update_state(
                state='PROGRESS',
                meta={
                    'step': f'Generated {len(chart_paths)} charts',
                    'progress': 90,
                    'status': 'processing',
                    'charts_generated': len(chart_paths)
                }
            )
            
            # Convert Path objects to strings for JSON serialization
            chart_paths_str = [str(p) for p in chart_paths]
            
            # Return success result
            result = {
                'success': True,
                'charts': chart_paths_str,
                'summary': {
                    'nwea': {
                        'rows': len(nwea_data),
                        'columns': len(nwea_data[0]) if nwea_data else 0
                    }
                },
                'temp_charts_dir': temp_charts_dir,  # Keep temp dir for now
                'step': 'Complete!',
                'progress': 100,
                'status': 'completed'
            }
            
            return result
            
        except Exception as e:
            # Clean up temp directory on error
            if temp_charts_dir and Path(temp_charts_dir).exists():
                try:
                    shutil.rmtree(temp_charts_dir)
                except:
                    pass
            raise e
            
    except Exception as e:
        error_msg = str(e)
        import traceback
        traceback.print_exc()
        
        # Update task state to FAILURE
        self.update_state(
            state='FAILURE',
            meta={
                'error': error_msg,
                'step': 'Failed',
                'progress': 0,
                'status': 'failed'
            }
        )
        
        return {
            'success': False,
            'error': error_msg,
            'step': 'Failed',
            'progress': 0,
            'status': 'failed'
        }

