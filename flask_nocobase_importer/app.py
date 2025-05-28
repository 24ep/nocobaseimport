from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify # Added jsonify
import os
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np 
import json 
import traceback # Add to imports
from flask_talisman import Talisman # Import Talisman
from flask_rq2 import RQ # Add this import
from rq.job import Job # Add this import
from rq.exceptions import NoSuchJobError # Add this import

from flask_nocobase_importer.db_manager import DatabaseManager
from flask_nocobase_importer.data_processing import (
    get_collections, read_excel_file, clean_dataframe, # process_dependency will be called by the task
    make_position_at_work, make_extension, get_field_definitions, # These are used by the task
    # resolve_extra_columns, validate_dataframe, preprocess_foreign_keys, (these are called by validation task)
    # preprocess_array_fields, drop_relationship_fields, (these are called by validation task)
    # upload_data will be called by the upload task
)
from .tasks import run_dependency_processing_task, run_validation_task, run_upload_task # Import the new task

app = Flask(__name__)

# Initialize Talisman with default settings for security headers
# The default CSP is quite strict:
# default-src 'self'; object-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; frame-ancestors 'self';
# You might need to adjust this if you have external resources or inline scripts/styles.
# For now, the default is a good starting point.
talisman = Talisman(app)

# Use environment variables for configuration
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "a_secure_default_dev_key")
app.config['UPLOAD_FOLDER'] = os.environ.get("FLASK_UPLOAD_FOLDER", "uploads/")

# Secure Session Cookie Settings
# Note: SESSION_COOKIE_SECURE should ideally be True if the app is served over HTTPS.
# If deploying behind a reverse proxy that handles TLS, ensure it sets appropriate headers
# like X-Forwarded-Proto, and that Flask is configured to trust them (e.g. via ProxyFix middleware).
# For now, we'll set it based on an environment variable, defaulting to False for easier local HTTP dev.
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_SESSION_COOKIE_SECURE', 'False').lower() == 'true'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Set a maximum content length for file uploads (e.g., 16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Configure Flask-RQ2
app.config['RQ_REDIS_URL'] = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
app.config['RQ_QUEUE_CLASS'] = 'rq.Queue' # Optional: Explicitly set queue class
rq = RQ(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    db = DatabaseManager() 
    try:
        collections_list = get_collections(db)
    except Exception as e:
        flash(f"Error connecting to database or fetching collections: {e}", "error")
        collections_list = []
    # Session data is now cleared by /reset_and_start or by the global error handler
    # db = DatabaseManager() 
    # try:
    #     collections_list = get_collections(db)
    # except Exception as e:
    #     flash(f"Error connecting to database or fetching collections: {e}", "error")
    #     collections_list = []
    # return render_template('index.html', title='NocoBase Importer - Step 1', collections=collections_list)

@app.route('/reset_and_start')
def reset_and_start():
    # Clear all relevant session keys
    keys_to_clear = [
        'uploaded_filepath', 'selected_collection', 'processed_df_path',
        'dependency_statuses', 'all_dependencies_processed_successfully',
        'passed_df_path', 'error_df_path', 'unresolved_values_json'
    ]
    for key in keys_to_clear:
        session.pop(key, None)
    flash("Session cleared. Starting a new import.", "info")
    return redirect(url_for('index'))

@app.route('/')
def index():
    db = DatabaseManager() 
    try:
        collections_list = get_collections(db)
    except Exception as e:
        flash(f"Error connecting to database or fetching collections: {e}", "error")
        collections_list = []
    # Clear all session data relevant to a previous import run
    # This is now handled by reset_and_start, but keeping it here for direct navigation to / might be desired.
    # However, the primary "start new" button now goes to /reset_and_start.
    # For robustness, if user lands on / directly, we can still clear.
    # Alternatively, remove this clear from index() if all start-over paths go through reset_and_start.
    # Let's assume for now that reset_and_start is the primary way to clear.
    # session.pop('uploaded_filepath', None)
    # session.pop('selected_collection', None)
    session.pop('processed_df_path', None)
    session.pop('dependency_statuses', None)
    session.pop('all_dependencies_processed_successfully', None)
    session.pop('passed_df_path', None)
    session.pop('error_df_path', None)
    session.pop('unresolved_values_json', None)
    return render_template('index.html', title='Step 1: Upload & Configure', collections=collections_list)

@app.route('/upload_config', methods=['POST'])
def upload_config():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))
    
    selected_collection = request.form.get('collection')
    if not selected_collection:
        flash('No collection selected', 'error')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            session['uploaded_filepath'] = filepath
            session['selected_collection'] = selected_collection

            df = read_excel_file(filepath) 
            df = clean_dataframe(df)
            
            df_filename = f"{secure_filename(selected_collection)}_initial_processed_df.parquet"
            df_path = os.path.join(app.config['UPLOAD_FOLDER'], df_filename)
            df.to_parquet(df_path)
            session['processed_df_path'] = df_path
            
            flash(f"File '{filename}' uploaded successfully for collection '{selected_collection}'.", "success")
            return redirect(url_for('process_dependencies'))

        except Exception as e:
            flash(f"Error processing file: {e}", "error")
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError as ose: 
                    app.logger.error(f"Error removing file {filepath}: {ose}")
            session.pop('uploaded_filepath', None)
            session.pop('selected_collection', None)
            session.pop('processed_df_path', None)
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Allowed types are .xlsx, .xls', 'error')
        return redirect(url_for('index'))

@app.route('/process_dependencies', methods=['GET'])
def process_dependencies():
    processed_df_path = session.get('processed_df_path')
    selected_collection = session.get('selected_collection')

    if not processed_df_path or not selected_collection:
        flash("Initial data not processed or session expired. Please start from Step 1.", "error")
        return redirect(url_for('index'))

    if not os.path.exists(processed_df_path):
        flash(f"Processed data file not found at {processed_df_path}. Please re-upload.", "error")
        session.pop('processed_df_path', None) 
        session.pop('selected_collection', None)
        return redirect(url_for('index'))
        
    try:
        df = pd.read_parquet(processed_df_path)
    except Exception as e:
        flash(f"Error reading processed data: {e}. Please re-upload.", "error")
        return redirect(url_for('index'))

    db = DatabaseManager()
    session['dependency_statuses'] = {} # This will be populated by the task results eventually

    # Enqueue the background task
    try:
        # Use the default queue, ensure rq is initialized (it is globally in this file)
        queue = rq.get_queue()
        meta = {'job_type': 'dependency_processing'} # Add job_type to meta
        job = queue.enqueue(
            run_dependency_processing_task,
            args=(processed_df_path, selected_collection),
            job_timeout='2h', # Example timeout
            result_ttl=86400, # Keep result for 1 day
            meta=meta # Pass meta to the job
        )
        session['dependency_job_id'] = job.id
        flash(f"Dependency processing started in the background (Job ID: {job.id}). You will be redirected to the progress page.", "info")
        return redirect(url_for('import_job_progress', job_id=job.id))
    except Exception as e:
        flash(f"Error enqueuing dependency processing task: {e}", "error")
        app.logger.error(f"Failed to enqueue task: {e}", exc_info=True)
        return redirect(url_for('index'))

    # The old synchronous logic is now moved to tasks.py
    # if selected_collection == "visitor_information":
    #     app.logger.info("Performing visitor_information specific preprocessing.")
        if "company_name_en" in df.columns or "company_name_th" in df.columns:
            df["lookup_company"] = df.get("company_name_en", pd.Series(dtype='object')).fillna("").replace("", pd.NA)
            df["lookup_company"] = df["lookup_company"].fillna(df.get("company_name_th", pd.Series(dtype='object')).fillna(""))
            df["lookup_company"] = df["lookup_company"].replace("", pd.NA)
        else:
            app.logger.warning("Skipping lookup_company: no 'company_name_en' or 'company_name_th' column found.")
            if "lookup_company" not in df.columns : 
                df["lookup_company"] = pd.NA

        if {"lookup_position", "lookup_department"}.issubset(df.columns) and "lookup_company" in df.columns:
             df['position_at_work'] = df.apply(make_position_at_work, axis=1)
        else:
            app.logger.warning("Skipping position_at_work: missing one or more of 'lookup_position', 'lookup_department', 'lookup_company'.")

        for i in [1, 2]:
            num_col = f"telephone_no_{i}"
            ext_col = f"extension_{i}"
            out_col = f"telephone_extension_{i}"
            if num_col in df.columns or ext_col in df.columns:
                df[out_col] = df.apply(lambda row: make_extension(row, num_col, ext_col), axis=1)
            
        phone_cols = [c for c in ["mobile_no_1", "mobile_no_2", "telephone_extension_1", "telephone_extension_2"] if c in df.columns]
        if phone_cols:
            df['phone_number'] = df[phone_cols].astype(str).agg(
                lambda x: '|'.join(i for i in x.dropna() if str(i).strip() and str(i).lower() not in ['nan', '<na>', 'none']), axis=1 
            ).replace('', pd.NA) 
        else:
            app.logger.warning("Skipping phone_number: no source phone columns found.")
        
        try:
            df.to_parquet(processed_df_path) 
            app.logger.info(f"DataFrame for '{selected_collection}' updated with preprocessing and saved to {processed_df_path}")
        except Exception as e:
            flash(f"Error saving preprocessed data for visitor_information: {e}", "error")
            return redirect(url_for('index'))

    dependencies = {}
    if "origin_source" in df.columns:
        dependencies["original_source"] = {
            "cols": ["origin_source"], "mapping": {"origin_source": "source_name"}
        }
    if "email" in df.columns:
        dependencies["email_list"] = {
            "cols": ["email"], "mapping": {"email": "email_list"}
        }
    
    if "phone_number" in df.columns:
        df_phone_exploded = df.copy()
        if not df_phone_exploded["phone_number"].empty and df_phone_exploded["phone_number"].dropna().apply(lambda x: isinstance(x, str) and "|" in x).any():
             df_phone_exploded["phone_number"] = df_phone_exploded["phone_number"].astype(str).str.split("|")
             df_phone_exploded = df_phone_exploded.explode("phone_number").reset_index(drop=True)
             df_phone_exploded["phone_number"] = df_phone_exploded["phone_number"].str.strip().replace('', pd.NA) 
        dependencies["calling_list"] = {
            "cols": ["phone_number"], "mapping": {"phone_number": "number_with_extension"},
            "source_df": df_phone_exploded 
        }

    if "lookup_company" in df.columns:
        df_company_exploded = df.copy()
        if not df_company_exploded["lookup_company"].empty and df_company_exploded["lookup_company"].dropna().apply(lambda x: isinstance(x, str) and "|" in x).any():
            df_company_exploded["lookup_company"] = df_company_exploded["lookup_company"].astype(str).str.split("|")
            df_company_exploded = df_company_exploded.explode("lookup_company").reset_index(drop=True)
            df_company_exploded["lookup_company"] = df_company_exploded["lookup_company"].str.strip().replace('',pd.NA)
        company_cols = ["lookup_company", "company_name_en", "company_name_th", "company_email",
                        "company_website", "company_facebook", "company_register_capital",
                        "company_employee_no", "company_product_profile",
                        "lookup_industry", "lookup_sub_industry"]
        existing_company_cols = [col for col in company_cols if col in df_company_exploded.columns]
        dependencies["company"] = {
            "cols": existing_company_cols,
            "mapping": {"lookup_company": "company_name_code", 
                        "company_register_capital":"register_capital",
                        "company_employee_no":"company_employee_size",
                        "lookup_industry":"industry", "lookup_sub_industry":"sub_industry"},
            "source_df": df_company_exploded
        }

    if "position_at_work" in df.columns:
        df_paw_exploded = df.copy()
        if not df_paw_exploded["position_at_work"].empty and df_paw_exploded["position_at_work"].dropna().apply(lambda x: isinstance(x, str) and "|" in x).any():
            df_paw_exploded["position_at_work"] = df_paw_exploded["position_at_work"].astype(str).str.split("|")
            df_paw_exploded = df_paw_exploded.explode("position_at_work").reset_index(drop=True)
            df_paw_exploded["position_at_work"] = df_paw_exploded["position_at_work"].str.strip().replace('',pd.NA)
        paw_cols = ["position_at_work", "lookup_position", "lookup_department", "lookup_company",
                    "lookup_industry", "lookup_sub_industry"]
        existing_paw_cols = [col for col in paw_cols if col in df_paw_exploded.columns]
        dependencies["position_at_work"] = {
            "cols": existing_paw_cols,
            "mapping": {"lookup_position":"position", "lookup_department":"department", 
                        "lookup_company":"company"},
             "source_df": df_paw_exploded
        }

    dependency_statuses = {}
    all_success = True
    if not dependencies:
        flash("No dependencies to process for this collection based on available columns.", "info")
    else:
        for dep_name, config in dependencies.items():
            app.logger.info(f"Processing dependency: {dep_name}")
            df_for_processing = config.get("source_df", df).copy() 
            try:
                success, error_df, count = process_dependency(db, dep_name, config, df_for_processing)
                error_json = None
                if error_df is not None and not error_df.empty:
                     error_json = error_df.to_json(orient='split', index=False) 
                     all_success = False if not success else all_success 
                dependency_statuses[dep_name] = {'success': success, 'count': count, 'errors': error_json}
                if not success:
                    all_success = False
                    flash(f"Dependency '{dep_name}' failed to process fully.", "warning")
                elif error_json:
                    flash(f"Dependency '{dep_name}' processed with {count} upserts, but some source rows had validation issues.", "warning")
                else:
                    flash(f"Dependency '{dep_name}' processed successfully ({count} rows).", "success")
            except Exception as e:
                app.logger.error(f"Critical error processing dependency {dep_name}: {e}", exc_info=True)
                dependency_statuses[dep_name] = {'success': False, 'count': 0, 'errors': f"Critical error: {e}"}
                all_success = False
                flash(f"A critical error occurred while processing dependency '{dep_name}'.", "error")
    
    session['dependency_statuses'] = dependency_statuses
    # session['all_dependencies_processed_successfully'] = all_success

    # return render_template('process_dependencies.html', 
    #                        title='Step 2: Process Dependencies',
    #                        selected_collection=selected_collection,
    #                        dependency_statuses=dependency_statuses,
    #                        all_success=all_success)

@app.route('/import_job/<job_id>')
def import_job_progress(job_id):
    try:
        job = rq.get_queue().fetch_job(job_id) # Use rq instance
        if job is None:
            flash(f"Job {job_id} not found or may have expired.", "error")
            return redirect(url_for('index')) # Or a dedicated page for job history/errors
        
        # Initial data for the template (polling will update it)
        status = job.get_status()
        meta = job.meta or {}
        
        return render_template('import_job_progress.html',
                               job_id=job.id,
                               status=status,
                               meta=meta, # Pass whole meta for flexibility
                               title=f"Job {job.id} Progress")
    except NoSuchJobError:
        flash(f"Job {job_id} not found (NoSuchJobError).", "error")
        return redirect(url_for('index'))
    except Exception as e:
        app.logger.error(f"Error fetching job for progress page {job_id}: {e}", exc_info=True)
        flash("An error occurred while trying to display job progress.", "error")
        return redirect(url_for('index'))

@app.route('/import_job_status/<job_id>')
def import_job_status(job_id):
    try:
        job = rq.get_queue().fetch_job(job_id) # Use rq instance
        if job is None:
            return jsonify({'status': 'NOT_FOUND', 'error': 'Job not found or may have expired.'}), 404
        
        status = job.get_status()
        meta = job.meta or {}
        response_data = {
            'job_id': job.id,
            'status': status,
            'is_finished': job.is_finished,
            'is_failed': job.is_failed,
            'progress': meta.get('progress', 0),
            'logs': meta.get('logs', []),
            'results': meta.get('results', None),
            'error_message': None, # Default to None
            'job_type': meta.get('job_type', 'unknown') 
        }
        if job.is_failed:
            # job.exc_info can be a long string with traceback
            # Provide a summary or a specific part if needed
            response_data['error_message'] = str(job.exc_info) if job.exc_info else "Job failed without detailed error information."
            # Ensure progress is 100 on failure to stop client-side polling correctly if it relies on progress only
            # However, is_failed should be the primary flag for stopping.
            # response_data['progress'] = 100 


        return jsonify(response_data)
    except NoSuchJobError:
        return jsonify({'status': 'NOT_FOUND', 'error': 'Job not found (NoSuchJobError)'}), 404
    except Exception as e:
        app.logger.error(f"Error fetching job status for {job_id}: {e}", exc_info=True)
        return jsonify({'status': 'ERROR', 'error': 'Failed to retrieve job status'}), 500

@app.route('/validate_data')
def validate_data():
    # Check if dependency processing was successful (if a job was run)
    dependency_job_id = session.get('dependency_job_id')
    if dependency_job_id:
        job = rq.get_queue().fetch_job(dependency_job_id)
        if job:
            if job.get_status() == 'finished':
                # Retrieve results from job.meta if stored there by the task
                # The task now stores results in job.meta['results']
                # And final status in job.meta['status'] ('SUCCESS', 'PARTIAL_FAILURE', 'FAILURE')
                task_final_status = job.meta.get('status')
                if task_final_status == 'SUCCESS':
                    flash("Dependency processing completed successfully.", "success")
                elif task_final_status == 'PARTIAL_FAILURE':
                     flash("Dependency processing completed with some failures. Check job logs for details.", "warning")
                elif task_final_status == 'FAILURE':
                    flash("Dependency processing failed. Check job logs. Validation cannot proceed.", "error")
                    return redirect(url_for('import_job_progress', job_id=dependency_job_id)) # Prevent proceeding
                # Store results in session if needed for other purposes, but all_dependencies_processed_successfully is less critical now
                session['dependency_statuses'] = job.meta.get('results', {}) 
                # session['all_dependencies_processed_successfully'] = (task_final_status == 'SUCCESS') # Redundant if we gatekeep on FAILURE

            elif job.get_status() not in ['finished', 'failed']:
                flash("Dependency processing is still in progress. Please wait.", "info")
                return redirect(url_for('import_job_progress', job_id=dependency_job_id))
            else: # job failed (e.g. worker died, not just task returning failure status in meta)
                flash(f"Dependency processing job {dependency_job_id} failed critically. Please review job details before proceeding.", "error")
                return redirect(url_for('import_job_progress', job_id=dependency_job_id)) # Prevent proceeding
        else: # Job not found in RQ (e.g. expired)
            flash(f"Could not retrieve status for dependency job {dependency_job_id}. Validation cannot proceed.", "error")
            return redirect(url_for('index')) # Or some other sensible default
    
    # If no dependency_job_id, implies it was a collection without dependencies, or direct navigation (less ideal)
    # The check for session.get('all_dependencies_processed_successfully') is now removed as it's less reliable than job status.

    # If we reach here, it means:
    # 1. There was a dependency_job_id, it finished, and its status was not 'FAILURE'.
    # 2. There was no dependency_job_id (e.g., no dependencies defined for the collection).
    processed_df_path = session.get('processed_df_path')
    selected_collection = session.get('selected_collection')

    if not processed_df_path or not selected_collection:
        flash("Session data missing for validation. Please start from Step 1.", "error")
        return redirect(url_for('index'))

    if not os.path.exists(processed_df_path):
        flash(f"Processed data file for validation not found at {processed_df_path}. Please re-upload.", "error")
        return redirect(url_for('index'))

    try:
        # Enqueue the validation task
        queue = rq.get_queue()
        validation_job = queue.enqueue(
            run_validation_task,
            args=(processed_df_path, selected_collection, app.config['UPLOAD_FOLDER']),
            job_timeout='2h', # Example timeout for validation
            result_ttl=86400, # Keep result for 1 day
            meta={'job_type': 'validation'}
        )
        session['validation_job_id'] = validation_job.id
        flash(f"Data validation started in the background (Job ID: {validation_job.id}). You will be redirected to the progress page.", "info")
        return redirect(url_for('import_job_progress', job_id=validation_job.id))
    except Exception as e:
        flash(f"Error enqueuing data validation task: {e}", "error")
        app.logger.error(f"Failed to enqueue validation task: {e}", exc_info=True)
        # If enqueuing fails, redirect back to dependency progress or index
        dep_job_id = session.get('dependency_job_id')
        if dep_job_id:
            return redirect(url_for('import_job_progress', job_id=dep_job_id))
        return redirect(url_for('index'))


@app.route('/upload_data_form')
def upload_data_form():
    validation_job_id = session.get('validation_job_id')
    selected_collection = session.get('selected_collection') # Keep selected_collection for title

    if not validation_job_id:
        flash("Validation job not started or session expired. Please start from Step 3 (Validate Data).", "error")
        return redirect(url_for('validate_data')) # Redirect to trigger validation

    try:
        job = rq.get_queue().fetch_job(validation_job_id)
        if job is None:
            flash(f"Validation job {validation_job_id} not found. Please try validation again.", "error")
            return redirect(url_for('validate_data'))

        if job.is_finished:
            job_meta_status = job.meta.get('status')
            if job_meta_status == 'SUCCESS':
                results = job.meta.get('results', {})
                passed_df_filename = results.get('passed_df_filename')
                passed_df_count = results.get('passed_df_count', 0)
                
                if not passed_df_filename:
                    flash("Validation completed, but passed data file information is missing.", "error")
                    return redirect(url_for('import_job_progress', job_id=validation_job_id))

                # Store filename in session to construct path in /execute_upload
                session['passed_df_filename'] = passed_df_filename 
                session['passed_df_count'] = passed_df_count # For display on form

                if passed_df_count == 0:
                    flash("Validation complete, but no data passed. Cannot proceed to upload.", "warning")
                    # Render form but it will show the 'passed_df_empty' message
                else:
                     flash("Validation successful. Ready for upload.", "success")
            
            elif job_meta_status == 'FAILURE':
                flash("Validation task failed. Please review job logs. Cannot proceed to upload.", "error")
                return redirect(url_for('import_job_progress', job_id=validation_job_id))
            else: # Other statuses like PARTIAL_FAILURE if you implement it
                flash(f"Validation task completed with status: {job_meta_status}. Review logs. Proceed with caution.", "warning")
                # Potentially allow proceeding, or redirect to progress page. For now, let's allow.
                results = job.meta.get('results', {})
                session['passed_df_filename'] = results.get('passed_df_filename')
                session['passed_df_count'] = results.get('passed_df_count',0)


        elif job.is_failed:
            flash("Validation task failed critically. Cannot proceed to upload.", "error")
            return redirect(url_for('import_job_progress', job_id=validation_job_id))
        else: # Job is still running or queued
            flash("Validation is still in progress. Please wait.", "info")
            return redirect(url_for('import_job_progress', job_id=validation_job_id))

    except NoSuchJobError:
        flash(f"Validation job {validation_job_id} not found (NoSuchJobError). Please try validation again.", "error")
        return redirect(url_for('validate_data'))
    except Exception as e:
        flash(f"Error retrieving validation job status: {e}", "error")
        app.logger.error(f"Error fetching validation job {validation_job_id} for upload_data_form: {e}", exc_info=True)
        return redirect(url_for('validate_data'))
    
    # Fetch collection columns for the form dropdowns
    db = DatabaseManager()
    collection_columns = []
    if selected_collection:
        try:
            field_defs = get_field_definitions(db, selected_collection)
            collection_columns = list(field_defs.keys())
        except Exception as e:
            flash(f"Error fetching field definitions for collection '{selected_collection}': {e}", "error")


    return render_template('upload_data_form.html',
                           title='Step 4: Select Upload Options',
                           selected_collection=selected_collection, # Ensure selected_collection is passed
                           collection_columns=collection_columns,
                           passed_df_empty=(session.get('passed_df_count', 0) == 0))


@app.route('/execute_upload', methods=['POST'])
def execute_upload():
    passed_df_filename = session.get('passed_df_filename') # Use filename from session
    selected_collection = session.get('selected_collection')

    if not passed_df_filename or not selected_collection:
        flash("Session data for upload missing or expired. Please start from Step 1.", "error")
        return redirect(url_for('index'))
    
    passed_df_path = os.path.join(app.config['UPLOAD_FOLDER'], passed_df_filename) # Still need this for path construction logic

    if not os.path.exists(passed_df_path): # Check if source file exists before enqueuing
        flash(f"Passed data file '{passed_df_filename}' not found. Please re-validate.", "error")
        return redirect(url_for('validate_data'))

    upload_mode = request.form.get('upload_mode')
    pk_column = request.form.get('pk_column') if upload_mode == 'update' else None
    conflict_column = request.form.get('conflict_column') if upload_mode == 'insert on duplicate update' else None
    
    # Basic validation for conditional fields
    if upload_mode == 'update' and not pk_column:
        flash("Primary Key column must be selected for 'update' mode.", "error")
        return redirect(url_for('upload_data_form'))
    if upload_mode == 'insert on duplicate update' and not conflict_column:
        flash("Conflict Column must be selected for 'insert on duplicate update' mode.", "error")
        return redirect(url_for('upload_data_form'))
    
    try:
        queue = rq.get_queue()
        upload_job = queue.enqueue(
            run_upload_task,
            args=(passed_df_filename, selected_collection, upload_mode, pk_column, conflict_column, app.config['UPLOAD_FOLDER']),
            job_timeout='3h', # Example timeout for upload
            result_ttl=86400, # Keep result for 1 day
            meta={'job_type': 'upload'}
        )
        session['upload_job_id'] = upload_job.id
        flash("Final data upload has started in the background. You will be redirected to the progress page.", "info")
        return redirect(url_for('import_job_progress', job_id=upload_job.id))

    except Exception as e:
        flash(f"Error enqueuing data upload task: {e}", "error")
        app.logger.error(f"Failed to enqueue upload task: {e}", exc_info=True)
        return redirect(url_for('upload_data_form')) # Redirect back to form on enqueue failure


@app.route('/upload_status')
def upload_status():
    return render_template('upload_status.html', title='Upload Status')


if __name__ == '__main__':
    port = int(os.environ.get("FLASK_RUN_PORT", 5000))
    host = os.environ.get("FLASK_RUN_HOST", "0.0.0.0") 
    debug_mode = os.environ.get("FLASK_DEBUG", "True").lower() == "true"
    app.run(host=host, port=port, debug=debug_mode)

@app.errorhandler(Exception)
def handle_unexpected_error(e):
    tb_str = traceback.format_exc()
    app.logger.error(f"Unhandled exception: {e}\n{tb_str}")
    flash(f"An unexpected error occurred: {e}. Please try again or contact support if the issue persists.", "error")
    # Clear potentially problematic session data to allow a fresh start
    keys_to_clear = [
        'uploaded_filepath', 'selected_collection', 'processed_df_path',
        'dependency_statuses', 'all_dependencies_processed_successfully',
        'passed_df_path', 'error_df_path', 'unresolved_values_json'
    ]
    for key in keys_to_clear:
        session.pop(key, None)
    return redirect(url_for('index'))

# Commented out Streamlit code remains below...
# ... (rest of the commented code) ...
