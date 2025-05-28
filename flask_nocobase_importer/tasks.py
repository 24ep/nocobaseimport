import os 
import pandas as pd
import logging
import json
from rq import get_current_job
# from flask import current_app # Not strictly needed if MinioManager handles its own config

from flask_nocobase_importer.db_manager import DatabaseManager
from flask_nocobase_importer.data_processing import (
    process_dependency, validate_dataframe, upload_data,
    get_field_definitions, resolve_extra_columns, 
    preprocess_foreign_keys, preprocess_array_fields,
    drop_relationship_fields, make_position_at_work, make_extension,
    clean_dataframe # Added clean_dataframe
)
from flask_nocobase_importer.minio_manager import MinioManager 
from werkzeug.utils import secure_filename 
import traceback 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Helper function to log progress and update job meta
def _log_task_progress(job, message, percentage, status_override=None):
    current_status = job.meta.get('status', 'STARTED') # Default to 'STARTED' if not set
    if status_override:
        current_status = status_override
    
    log_entry = f"[{pd.Timestamp.now()}] {message}"
    job.meta['logs'].append(log_entry)
    job.meta['progress'] = percentage
    job.meta['status'] = current_status # Set the status, possibly overridden
    logger.info(log_entry) # Also log to standard logger
    job.save_meta()


def run_dependency_processing_task(processed_df_path, selected_collection):
    job = get_current_job()
    _log_task_progress(job, "Dependency processing task started.", 0, status_override='STARTED')

    if not os.path.exists(processed_df_path):
        _log_task_progress(job, f"Error: Processed DataFrame file not found at {processed_df_path}.", 0, status_override='FAILURE')
        job.meta['results'] = {'error': "Processed DataFrame file not found."}
        job.save_meta()
        return "Task failed: Processed DataFrame file not found."

    try:
        df = pd.read_parquet(processed_df_path)
        _log_task_progress(job, f"Loaded initial DataFrame. Shape: {df.shape}", 5)
        
        db = DatabaseManager()
        _log_task_progress(job, "DatabaseManager initialized.", 10)

        # --- Visitor Information Specific Preprocessing (if applicable) ---
        # This logic was originally in app.py, moved here for background processing
        if selected_collection == "visitor_information":
            _log_task_progress(job, "Performing 'visitor_information' specific preprocessing.", 15)
            if "company_name_en" in df.columns or "company_name_th" in df.columns:
                df["lookup_company"] = df.get("company_name_en", pd.Series(dtype='object')).fillna("").replace("", pd.NA)
                df["lookup_company"] = df["lookup_company"].fillna(df.get("company_name_th", pd.Series(dtype='object')).fillna(""))
                df["lookup_company"] = df["lookup_company"].replace("", pd.NA)
                _log_task_progress(job, "Processed 'lookup_company'.", 20)
            else:
                if "lookup_company" not in df.columns : 
                    df["lookup_company"] = pd.NA # Ensure column exists if not created
                _log_task_progress(job, "Skipped 'lookup_company': source columns not found.", 20)

            # Aligning with Streamlit's condition for creating 'position_at_work'
            # The make_position_at_work function handles missing lookup_company internally.
            if {"lookup_position", "lookup_department"}.issubset(df.columns):
                df['position_at_work'] = df.apply(make_position_at_work, axis=1)
                _log_task_progress(job, "Processed 'position_at_work'.", 25)
            else:
                _log_task_progress(job, "Skipped 'position_at_work': missing 'lookup_position' or 'lookup_department'.", 25)
            
            for i in [1, 2]:
                num_col = f"telephone_no_{i}"
                ext_col = f"extension_{i}"
                out_col = f"telephone_extension_{i}"
                if num_col in df.columns or ext_col in df.columns:
                    df[out_col] = df.apply(lambda row: make_extension(row, num_col, ext_col), axis=1)
                    _log_task_progress(job, f"Processed '{out_col}'.", 30 + (i*2))
            
            phone_cols = [c for c in ["mobile_no_1", "mobile_no_2", "telephone_extension_1", "telephone_extension_2"] if c in df.columns]
            if phone_cols:
                df['phone_number'] = df[phone_cols].astype(str).agg(
                    lambda x: '|'.join(i for i in x.dropna() if str(i).strip() and str(i).lower() not in ['nan', '<na>', 'none']), axis=1
                ).replace('', pd.NA)
                _log_task_progress(job, "Processed 'phone_number' aggregation.", 35)
            else:
                _log_task_progress(job, "Skipped 'phone_number' aggregation: no source columns.", 35)
            
            # Save the preprocessed DataFrame
            df.to_parquet(processed_df_path)
            _log_task_progress(job, f"Preprocessed DataFrame for '{selected_collection}' saved.", 40)
        # --- End Visitor Information Specific Preprocessing ---

        dependencies = {}
        # This logic for defining dependencies also came from app.py
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
        overall_success = True
        if not dependencies:
            _log_task_progress(job, "No dependencies to process for this collection.", 90, status_override='SUCCESS')
        else:
            total_deps = len(dependencies)
            current_dep_num = 0
            base_progress = 40 # Progress after specific preprocessing
            progress_per_dep = (90 - base_progress) / total_deps if total_deps > 0 else 0

            for dep_name, config in dependencies.items():
                current_dep_num += 1
                current_progress = base_progress + int(current_dep_num * progress_per_dep)
                _log_task_progress(job, f"Processing dependency: {dep_name} ({current_dep_num}/{total_deps})", current_progress)
                
                df_for_processing = config.get("source_df", df).copy()
                success, error_df, count = process_dependency(db, dep_name, config, df_for_processing)
                
                error_json = None
                if error_df is not None and not error_df.empty:
                    error_json = error_df.to_json(orient='records') # Changed to records for better readability if needed
                
                dependency_statuses[dep_name] = {'success': success, 'count': count, 'errors': error_json}
                
                if not success:
                    overall_success = False
                    _log_task_progress(job, f"Dependency '{dep_name}' processed with failures. Count: {count}. Errors: {error_json if error_json else 'N/A'}", current_progress)
                elif error_json: # Success is True, but there were errors (e.g. some rows failed validation within process_dependency)
                     _log_task_progress(job, f"Dependency '{dep_name}' processed with {count} upserts, but some source rows had issues (see errors).", current_progress)
                else:
                    _log_task_progress(job, f"Dependency '{dep_name}' processed successfully. Count: {count}.", current_progress)
        
        job.meta['results'] = dependency_statuses
        final_status = 'SUCCESS' if overall_success else 'PARTIAL_FAILURE'
        _log_task_progress(job, "All dependencies processed.", 100, status_override=final_status)
        return f"Dependency processing completed with status: {final_status}"

    except Exception as e:
        tb_str = traceback.format_exc()
        log_message = f"Critical error in dependency processing task: {e}\n{tb_str}"
        current_progress = job.meta.get('progress', 0) # Use current progress
        _log_task_progress(job, log_message, current_progress, status_override='FAILURE')
        job.meta['results'] = {'error': str(e), 'traceback': tb_str}
        job.save_meta()
        raise


def run_validation_task(processed_df_path, selected_collection, upload_folder_path):
    job = get_current_job()
    _log_task_progress(job, "Validation task started.", 0, status_override='STARTED')

    if not os.path.exists(processed_df_path):
        _log_task_progress(job, f"Error: Processed DataFrame file not found at {processed_df_path}.", 0, status_override='FAILURE')
        job.meta['results'] = {'error': "Processed DataFrame file not found."}
        job.save_meta()
        return "Task failed: Processed DataFrame file not found."

    try:
        df = pd.read_parquet(processed_df_path)
        _log_task_progress(job, f"Loaded DataFrame for validation. Shape: {df.shape}", 10)

        # 1. Initial clean (Streamlit does df.where(df != "", None), clean_dataframe is more comprehensive)
        # clean_dataframe replaces empty strings with pd.NA, then NAs (including original NaN) with None.
        # This is suitable.
        df = clean_dataframe(df)
        _log_task_progress(job, "Applied initial cleaning to DataFrame.", 15)

        db = DatabaseManager()
        _log_task_progress(job, "DatabaseManager initialized.", 20)
        
        # 2. Get field definitions
        field_definitions = get_field_definitions(db, selected_collection)
        _log_task_progress(job, f"Fetched field definitions for '{selected_collection}'.", 30)

        # 3. Resolve extra columns (operates on the main df)
        # resolve_extra_columns expects 2 args: df, field_defs
        # The third argument 'selected_collection' was removed from resolve_extra_columns in data_processing.py
        # in a previous step to align with Streamlit.
        df_resolved = resolve_extra_columns(df, field_definitions)
        _log_task_progress(job, "Resolved extra columns.", 40)

        # 4. Validate dataframe (operates on the resolved df)
        # validate_dataframe expects 3 args: df, field_defs, db
        # The fourth argument 'selected_collection' was removed from validate_dataframe in data_processing.py
        # in a previous step to align with Streamlit.
        passed_df, error_df, unresolved_values = validate_dataframe(df_resolved, field_definitions, db)
        _log_task_progress(job, f"Data validation complete. Passed: {len(passed_df)}, Errors: {len(error_df)}", 60)

        # 5. Preprocess foreign keys (operates on a copy of passed_df)
        # preprocess_foreign_keys expects 3 args: df, field_defs, db
        # The fourth argument 'selected_collection' was removed from preprocess_foreign_keys in data_processing.py
        # in a previous step to align with Streamlit.
        passed_df_processed = preprocess_foreign_keys(passed_df.copy(), field_definitions, db)
        _log_task_progress(job, "Preprocessed foreign keys for passed data.", 70)

        # 6. Preprocess array fields (operates on the already processed passed_df)
        passed_df_processed = preprocess_array_fields(passed_df_processed, field_definitions)
        _log_task_progress(job, "Preprocessed array fields for passed data.", 75)
        
        # 7. Drop relationship fields (operates on the already processed passed_df)
        passed_df_processed = drop_relationship_fields(passed_df_processed, field_definitions)
        _log_task_progress(job, "Dropped relationship fields from passed data.", 80)

        # 8. Final clean of the processed passed_df (NA to None)
        passed_df_final = passed_df_processed.where(pd.notna(passed_df_processed), None)
        _log_task_progress(job, "Applied final cleaning to passed data.", 85)

        # 9. Save passed and error DataFrames
        # (Filename generation and saving logic can remain similar but use passed_df_final)
        passed_df_filename = f"{secure_filename(selected_collection)}_passed_validation_df.parquet"
        error_df_filename = f"{secure_filename(selected_collection)}_failed_validation_df.parquet"
        
        passed_df_path = os.path.join(upload_folder_path, passed_df_filename)
        error_df_path = os.path.join(upload_folder_path, error_df_filename)
        
        passed_df_final.to_parquet(passed_df_path) # Save the fully processed passed_df
        error_df.to_parquet(error_df_path) # error_df comes directly from validate_dataframe
        _log_task_progress(job, "Saved passed and error DataFrames.", 95)

        job.meta['results'] = {
            'passed_df_filename': passed_df_filename,
            'passed_df_count': len(passed_df_final), # Use count from passed_df_final
            'error_df_filename': error_df_filename,
            'error_df_count': len(error_df),
            'unresolved_values_json': json.dumps(unresolved_values, ensure_ascii=False, indent=2) if unresolved_values else None
        }
        
        final_status = 'SUCCESS' if len(error_df) == 0 and len(unresolved_values) == 0 else 'PARTIAL_FAILURE' # Consider unresolved_values for status
        _log_task_progress(job, "Validation task completed.", 100, status_override=final_status)
        return f"Validation completed. Status: {final_status}"

    except Exception as e:
        tb_str = traceback.format_exc()
        log_message = f"Critical error in validation task: {e}\n{tb_str}"
        current_progress = job.meta.get('progress', 0)
        _log_task_progress(job, log_message, current_progress, status_override='FAILURE')
        job.meta['results'] = {'error': str(e), 'traceback': tb_str}
        job.save_meta()
        raise

# Update signature to include original_uploaded_filepath
def run_upload_task(passed_df_filename, selected_collection, upload_mode, pk_column, conflict_column, upload_folder_path, original_uploaded_filepath=None):
    job = get_current_job()

    # log_progress function definition (already exists, adapted to use _log_task_progress)
    def log_progress(message, percentage, status_override=None):
        _log_task_progress(job, message, percentage, status_override)


    job.meta['status'] = 'STARTED'
    job.meta['progress'] = 0
    job.meta['logs'] = ['Upload task started.']
    job.meta['results'] = {}
    job.save_meta()

    log_progress("Upload task initiated.", 0)
    final_pg_upload_status_message = "Upload task did not complete." # Default message

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
            log_progress("DataFrame is empty. Nothing to upload to PostgreSQL.", 80) # Progress before MinIO
            job.meta['results']['message'] = "PostgreSQL upload skipped (0 rows)."
            job.meta['results']['row_count'] = 0
            final_pg_upload_status_message = "PostgreSQL upload successful (0 rows)."
            # Proceed to MinIO backup even if DataFrame is empty, to backup the original empty template if needed.
        else:
            db = DatabaseManager()
            log_progress(f"Starting data upload to PostgreSQL collection '{selected_collection}' with mode '{upload_mode}'.", 30)
            upload_data(db, selected_collection, passed_df, upload_mode, pk=pk_column, conflict_col=conflict_column)
            row_count = len(passed_df)
            log_progress(f"PostgreSQL data upload successful. {row_count} rows processed.", 80) # Progress before MinIO
            job.meta['results']['message'] = f"Successfully uploaded {row_count} rows to PostgreSQL."
            job.meta['results']['row_count'] = row_count
            final_pg_upload_status_message = f"PostgreSQL upload successful ({row_count} rows)."

        # MinIO Backup Section
        if original_uploaded_filepath and os.path.exists(original_uploaded_filepath):
            log_progress(f"Attempting to back up original file '{original_uploaded_filepath}' to MinIO.", 85)
            minio_manager = MinioManager()
            if minio_manager.client: # Check if Minio client initialized successfully
                original_fn = os.path.basename(original_uploaded_filepath)
                # Sanitize collection name for path safety, though it should be safe from get_collections
                safe_collection_name = secure_filename(selected_collection)
                destination_filename_minio = f"{safe_collection_name}/original_uploads/{original_fn}"
                
                backup_success = minio_manager.upload_file(original_uploaded_filepath, destination_filename_minio)
                if backup_success:
                    log_progress(f"Successfully backed up '{original_fn}' to MinIO as '{destination_filename_minio}'.", 95)
                    job.meta['results']['minio_backup_status'] = 'Success'
                    job.meta['results']['minio_backup_destination'] = destination_filename_minio
                else:
                    log_progress(f"Failed to back up '{original_fn}' to MinIO. Check MinIO logs.", 95, status_override=job.meta['status']) # Don't change overall job status for backup failure
                    job.meta['results']['minio_backup_status'] = 'Failure'
            else:
                log_progress("MinIO client not initialized (check config/env vars). Backup skipped.", 95)
                job.meta['results']['minio_backup_status'] = 'Skipped - Client not initialized'
        elif original_uploaded_filepath:
            log_progress(f"Original uploaded file not found at '{original_uploaded_filepath}'. MinIO Backup skipped.", 95)
            job.meta['results']['minio_backup_status'] = 'Skipped - File not found'
        else:
            log_progress("No original uploaded file path provided. MinIO Backup skipped.", 95)
            job.meta['results']['minio_backup_status'] = 'Skipped - No path provided'
        
        # Final task status
        log_progress(f"{final_pg_upload_status_message}. MinIO backup status: {job.meta['results'].get('minio_backup_status', 'Not attempted')}.", 100, status_override='SUCCESS')
        return f"Upload task completed. {final_pg_upload_status_message}"

    except Exception as e:
        tb_str = traceback.format_exc()
        log_message = f"Critical error in upload task: {e}\n{tb_str}"
        # Use current progress or a safe default if progress not set yet
        current_progress = job.meta.get('progress', 0) 
        log_progress(log_message, current_progress, status_override='FAILURE')
        job.meta['results']['error'] = str(e)
        job.save_meta()
        raise
