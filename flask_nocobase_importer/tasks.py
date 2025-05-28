import os
import pandas as pd
import logging
import json
from rq import get_current_job
from flask import current_app # To access rq instance if needed, or app context

# Assuming DatabaseManager and other functions will be imported correctly
# This might require adjustments if app context is needed for DB or config
from flask_nocobase_importer.db_manager import DatabaseManager
from flask_nocobase_importer.data_processing import (
    get_field_definitions, make_position_at_work, make_extension, process_dependency,
    clean_dataframe # Added clean_dataframe as it's used in visitor_info preprocessing
)
# For run_validation_task
from flask_nocobase_importer.data_processing import (
    resolve_extra_columns, validate_dataframe, preprocess_foreign_keys,
    preprocess_array_fields, drop_relationship_fields, upload_data # Added upload_data
)
from werkzeug.utils import secure_filename # For saving files with secure names
import traceback # For detailed error logging

logger = logging.getLogger(__name__)
# Basic logging configuration for the task module, can be enhanced
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def run_dependency_processing_task(processed_df_path, selected_collection):
    job = get_current_job()
    
    def log_progress(message, percentage, status_override=None):
        current_status = job.meta.get('status', 'STARTED')
        if status_override:
            current_status = status_override
        
        log_entry = f"[{pd.Timestamp.now()}] {message}"
        job.meta['logs'].append(log_entry)
        job.meta['progress'] = percentage
        job.meta['status'] = current_status # Update status along with progress
        logger.info(log_entry) # Also log to worker console
        job.save_meta()

    job.meta['status'] = 'STARTED'
    job.meta['progress'] = 0
    job.meta['logs'] = []
    job.meta['results'] = {}
    job.save_meta()

    log_progress("Task started.", 0)

    try:
        if not os.path.exists(processed_df_path):
            log_progress(f"Error: Processed data file not found at {processed_df_path}.", 0, status_override='FAILURE')
            return f"Task failed: File not found {processed_df_path}"

        df = pd.read_parquet(processed_df_path)
        log_progress(f"Loaded DataFrame from {processed_df_path}. Shape: {df.shape}", 5)

        # It's better to get the RQ instance from current_app if tasks run within app context
        # For now, DatabaseManager is instantiated directly. If it needed app config, this would change.
        db = DatabaseManager()

        # visitor_information specific preprocessing
        if selected_collection == "visitor_information":
            log_progress("Performing visitor_information specific preprocessing.", 10)
            if "company_name_en" in df.columns or "company_name_th" in df.columns:
                df["lookup_company"] = df.get("company_name_en", pd.Series(dtype='object')).fillna("").replace("", pd.NA)
                df["lookup_company"] = df["lookup_company"].fillna(df.get("company_name_th", pd.Series(dtype='object')).fillna(""))
                df["lookup_company"] = df["lookup_company"].replace("", pd.NA)
            else:
                log_progress("Warning: Skipping lookup_company creation.", 12)
                if "lookup_company" not in df.columns:
                    df["lookup_company"] = pd.NA
            
            if {"lookup_position", "lookup_department"}.issubset(df.columns) and "lookup_company" in df.columns:
                 df['position_at_work'] = df.apply(make_position_at_work, axis=1)
            else:
                log_progress("Warning: Skipping position_at_work creation.", 14)

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
                log_progress("Warning: Skipping phone_number creation.", 18)
            
            # Save the preprocessed DataFrame back (important for subsequent steps)
            df = clean_dataframe(df) # Ensure it's cleaned before saving
            df.to_parquet(processed_df_path)
            log_progress("visitor_information preprocessing complete and DataFrame saved.", 20)
        
        dependencies = {}
        if "origin_source" in df.columns:
            dependencies["original_source"] = {"cols": ["origin_source"], "mapping": {"origin_source": "source_name"}}
        if "email" in df.columns:
            dependencies["email_list"] = {"cols": ["email"], "mapping": {"email": "email_list"}}
        
        if "phone_number" in df.columns:
            df_phone_exploded = df.copy()
            if not df_phone_exploded["phone_number"].empty and df_phone_exploded["phone_number"].dropna().apply(lambda x: isinstance(x, str) and "|" in x).any():
                df_phone_exploded["phone_number"] = df_phone_exploded["phone_number"].astype(str).str.split("|")
                df_phone_exploded = df_phone_exploded.explode("phone_number").reset_index(drop=True)
                df_phone_exploded["phone_number"] = df_phone_exploded["phone_number"].str.strip().replace('', pd.NA)
            dependencies["calling_list"] = {"cols": ["phone_number"], "mapping": {"phone_number": "number_with_extension"}, "source_df": df_phone_exploded}

        if "lookup_company" in df.columns:
            df_company_exploded = df.copy()
            if not df_company_exploded["lookup_company"].empty and df_company_exploded["lookup_company"].dropna().apply(lambda x: isinstance(x, str) and "|" in x).any():
                df_company_exploded["lookup_company"] = df_company_exploded["lookup_company"].astype(str).str.split("|")
                df_company_exploded = df_company_exploded.explode("lookup_company").reset_index(drop=True)
                df_company_exploded["lookup_company"] = df_company_exploded["lookup_company"].str.strip().replace('',pd.NA)
            company_cols = ["lookup_company", "company_name_en","company_name_th","company_email","company_website","company_facebook","company_register_capital","company_employee_no","company_product_profile","lookup_industry","lookup_sub_industry"]
            existing_company_cols = [col for col in company_cols if col in df_company_exploded.columns]
            dependencies["company"] = {"cols": existing_company_cols, "mapping": {"lookup_company": "company_name_code", "company_register_capital":"register_capital", "company_employee_no":"company_employee_size", "lookup_industry":"industry", "lookup_sub_industry":"sub_industry"}, "source_df": df_company_exploded}

        if "position_at_work" in df.columns:
            df_paw_exploded = df.copy()
            if not df_paw_exploded["position_at_work"].empty and df_paw_exploded["position_at_work"].dropna().apply(lambda x: isinstance(x, str) and "|" in x).any():
                df_paw_exploded["position_at_work"] = df_paw_exploded["position_at_work"].astype(str).str.split("|")
                df_paw_exploded = df_paw_exploded.explode("position_at_work").reset_index(drop=True)
                df_paw_exploded["position_at_work"] = df_paw_exploded["position_at_work"].str.strip().replace('',pd.NA)
            paw_cols = ["position_at_work", "lookup_position","lookup_department","lookup_company","lookup_industry","lookup_sub_industry"]
            existing_paw_cols = [col for col in paw_cols if col in df_paw_exploded.columns]
            dependencies["position_at_work"] = {"cols": existing_paw_cols, "mapping": {"lookup_position":"position", "lookup_department":"department", "lookup_company":"company"}, "source_df": df_paw_exploded}

        total_deps = len(dependencies)
        overall_success = True
        
        if not dependencies:
            log_progress("No dependencies defined or required for this collection.", 100, status_override='SUCCESS')
        else:
            initial_progress = job.meta['progress'] # Progress after visitor_info
            progress_per_dep = (100 - initial_progress) / total_deps if total_deps > 0 else 0

            for i, (dep_name, config) in enumerate(dependencies.items()):
                current_dep_progress_start = initial_progress + (i * progress_per_dep)
                log_progress(f"Processing dependency: {dep_name}", current_dep_progress_start)
                
                df_for_processing = config.get("source_df", df).copy()
                
                success, error_df, count = process_dependency(db, dep_name, config, df_for_processing)
                error_json = None
                if error_df is not None and not error_df.empty:
                    error_json = error_df.to_json(orient='split', index=False)
                
                job.meta['results'][dep_name] = {'success': success, 'count': count, 'errors': error_json}
                if not success:
                    overall_success = False
                
                current_dep_progress_end = initial_progress + ((i + 1) * progress_per_dep)
                log_progress(f"Finished dependency: {dep_name}. Success: {success}, Count: {count}", current_dep_progress_end)

        final_status = 'SUCCESS' if overall_success else 'PARTIAL_FAILURE' # Use PARTIAL_FAILURE if some deps failed
        log_progress("All dependencies processed.", 100, status_override=final_status)
        return f"Dependency processing finished with status: {final_status}"

    except Exception as e:
        tb_str = traceback.format_exc()
        log_message = f"Critical error in task: {e}\n{tb_str}"
        log_progress(log_message, job.meta.get('progress', 0), status_override='FAILURE')
        # Ensure results dict is present even on critical failure
        if 'results' not in job.meta:
            job.meta['results'] = {} 
        job.save_meta()
        # Re-raise to mark job as failed in RQ dashboard
        raise

def run_validation_task(processed_df_path, selected_collection, upload_folder_path):
    job = get_current_job()

    def log_progress(message, percentage, status_override=None):
        current_status = job.meta.get('status', 'STARTED')
        if status_override:
            current_status = status_override
        
        log_entry = f"[{pd.Timestamp.now()}] {message}"
        job.meta['logs'].append(log_entry)
        job.meta['progress'] = percentage
        job.meta['status'] = current_status
        logger.info(log_entry)
        job.save_meta()

    job.meta['status'] = 'STARTED'
    job.meta['progress'] = 0
    job.meta['logs'] = ['Validation task started.']
    job.meta['results'] = {} # Initialize results
    job.save_meta()

    log_progress("Validation task initiated.", 0)

    try:
        if not os.path.exists(processed_df_path):
            log_progress(f"Error: Processed DataFrame file not found at {processed_df_path}.", 0, status_override='FAILURE')
            return "Task failed: Source DataFrame file not found."

        df = pd.read_parquet(processed_df_path)
        log_progress(f"Loaded DataFrame. Shape: {df.shape}", 10)

        db = DatabaseManager() # Assumes DBManager can be instantiated without app context here
        field_defs = get_field_definitions(db, selected_collection)
        log_progress("Fetched field definitions.", 20)

        # Step 1: Resolve Extra Columns
        df_resolved = resolve_extra_columns(df.copy(), field_defs)
        log_progress("Resolved extra columns.", 30)

        # Step 2: Validate DataFrame
        # validate_dataframe returns: passed_df, error_df, unresolved_values
        df_for_further_processing, error_df_from_validation, unresolved_values = validate_dataframe(df_resolved.copy(), field_defs, db)
        log_progress("Validated DataFrame. Passed: {}, Errors: {}".format(len(df_for_further_processing), len(error_df_from_validation)), 50)

        # Step 3: Preprocess Foreign Keys (on the DataFrame that passed initial validation)
        passed_df_processed = preprocess_foreign_keys(df_for_further_processing.copy(), field_defs, db)
        log_progress("Preprocessed foreign keys.", 70)

        # Step 4: Preprocess Array Fields
        passed_df_processed = preprocess_array_fields(passed_df_processed.copy(), field_defs)
        log_progress("Preprocessed array fields.", 80)

        # Step 5: Drop Relationship Fields
        passed_df_processed = drop_relationship_fields(passed_df_processed.copy(), field_defs)
        log_progress("Dropped relationship fields.", 90)

        # Step 6: Final Cleaning
        passed_df_final = passed_df_processed.where(pd.notna(passed_df_processed), None)
        log_progress("Final cleaning complete.", 95)

        # Save DataFrames
        passed_df_filename = f"{secure_filename(selected_collection)}_passed_df.parquet"
        passed_df_filepath = os.path.join(upload_folder_path, passed_df_filename)
        passed_df_final.to_parquet(passed_df_filepath)
        job.meta['results']['passed_df_filename'] = passed_df_filename # Store filename
        job.meta['results']['passed_df_path_debug'] = passed_df_filepath # For debug, actual path

        error_df_filename = f"{secure_filename(selected_collection)}_error_df.parquet"
        error_df_filepath = os.path.join(upload_folder_path, error_df_filename)
        error_df_from_validation.to_parquet(error_df_filepath)
        job.meta['results']['error_df_filename'] = error_df_filename
        job.meta['results']['error_df_path_debug'] = error_df_filepath


        job.meta['results']['unresolved_values_json'] = json.dumps(unresolved_values if unresolved_values else {})
        job.meta['results']['passed_df_count'] = len(passed_df_final)
        job.meta['results']['error_df_count'] = len(error_df_from_validation)
        
        log_progress("Validation and processing complete. Files saved.", 100, status_override='SUCCESS')
        return "Validation task completed successfully."

    except Exception as e:
        tb_str = traceback.format_exc()
        log_message = f"Critical error in validation task: {e}\n{tb_str}"
        log_progress(log_message, job.meta.get('progress', 0), status_override='FAILURE')
        job.meta['results']['error'] = str(e) # Store error message in results
        job.save_meta()
        raise # Re-raise to mark job as failed in RQ

def run_upload_task(passed_df_filename, selected_collection, upload_mode, pk_column, conflict_column, upload_folder_path):
    job = get_current_job()

    def log_progress(message, percentage, status_override=None): # Renamed from log_progress to avoid conflict if file is not reloaded
        current_status = job.meta.get('status', 'STARTED')
        if status_override:
            current_status = status_override
        
        log_entry = f"[{pd.Timestamp.now()}] {message}"
        job.meta['logs'].append(log_entry)
        job.meta['progress'] = percentage
        job.meta['status'] = current_status
        logger.info(log_entry)
        job.save_meta()

    job.meta['status'] = 'STARTED'
    job.meta['progress'] = 0 # Can be simple 0 then 100 for this task
    job.meta['logs'] = ['Upload task started.']
    job.meta['results'] = {}
    job.save_meta()

    log_progress("Upload task initiated.", 0)

    try:
        passed_df_path = os.path.join(upload_folder_path, passed_df_filename)
        if not os.path.exists(passed_df_path):
            log_progress(f"Error: Passed DataFrame file not found at {passed_df_path}.", 0, status_override='FAILURE')
            job.meta['results']['error'] = "Passed DataFrame file not found."
            job.save_meta()
            return "Task failed: Passed DataFrame file not found."

        passed_df = pd.read_parquet(passed_df_path)
        log_progress(f"Loaded DataFrame for upload. Shape: {passed_df.shape}", 20)

        if passed_df.empty:
            log_progress("DataFrame is empty. Nothing to upload.", 100, status_override='SUCCESS') # Technically success, but 0 rows
            job.meta['results']['message'] = "Upload successful (0 rows)."
            job.meta['results']['row_count'] = 0
            job.save_meta()
            return "Upload successful (0 rows)."

        db = DatabaseManager() # Assumes DBManager can be instantiated without app context here

        log_progress(f"Starting data upload to collection '{selected_collection}' with mode '{upload_mode}'.", 30)
        
        # upload_data now raises exceptions on failure
        upload_data(db, selected_collection, passed_df, upload_mode, pk=pk_column, conflict_col=conflict_column)
        
        row_count = len(passed_df)
        log_progress(f"Data upload successful. {row_count} rows processed.", 100, status_override='SUCCESS')
        job.meta['results']['message'] = f"Successfully uploaded {row_count} rows."
        job.meta['results']['row_count'] = row_count
        job.save_meta()
        return f"Upload task completed successfully. {row_count} rows processed."

    except Exception as e:
        tb_str = traceback.format_exc()
        log_message = f"Critical error in upload task: {e}\n{tb_str}"
        log_progress(log_message, job.meta.get('progress', 0), status_override='FAILURE')
        job.meta['results']['error'] = str(e)
        job.save_meta()
        raise # Re-raise to mark job as failed in RQ
