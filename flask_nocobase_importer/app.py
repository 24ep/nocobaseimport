from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np 
import json 
import traceback # Add to imports

from flask_nocobase_importer.db_manager import DatabaseManager
from flask_nocobase_importer.data_processing import (
    get_collections, read_excel_file, clean_dataframe, process_dependency,
    make_position_at_work, make_extension, get_field_definitions,
    resolve_extra_columns, validate_dataframe, preprocess_foreign_keys, 
    preprocess_array_fields, drop_relationship_fields, upload_data # Added upload_data
)

app = Flask(__name__)
app.secret_key = 'dev_secret_key' 
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
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
    session['dependency_statuses'] = {} 

    if selected_collection == "visitor_information":
        app.logger.info("Performing visitor_information specific preprocessing.")
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
    session['all_dependencies_processed_successfully'] = all_success

    return render_template('process_dependencies.html', 
                           title='Step 2: Process Dependencies',
                           selected_collection=selected_collection,
                           dependency_statuses=dependency_statuses,
                           all_success=all_success)

@app.route('/validate_data')
def validate_data():
    processed_df_path = session.get('processed_df_path')
    selected_collection = session.get('selected_collection')

    if not processed_df_path or not selected_collection:
        flash("Session data missing or expired. Please start from Step 1.", "error")
        return redirect(url_for('index'))

    if not session.get('all_dependencies_processed_successfully', False) and session.get('dependency_statuses'):
        flash("Warning: Not all dependencies were processed successfully in the previous step. Validation results might be based on incomplete relational data.", "warning")

    if not os.path.exists(processed_df_path):
        flash(f"Processed data file not found at {processed_df_path}. Please re-upload.", "error")
        return redirect(url_for('index'))

    try:
        df = pd.read_parquet(processed_df_path)
        db = DatabaseManager()
        field_defs = get_field_definitions(db, selected_collection)
        
        df_resolved = resolve_extra_columns(df.copy(), field_defs)
        df_for_further_processing, error_df_from_validation, unresolved_values = validate_dataframe(df_resolved.copy(), field_defs, db)
        passed_df_processed = preprocess_foreign_keys(df_for_further_processing.copy(), field_defs, db)
        passed_df_processed = preprocess_array_fields(passed_df_processed.copy(), field_defs)
        passed_df_processed = drop_relationship_fields(passed_df_processed.copy(), field_defs)
        passed_df_final = passed_df_processed.where(pd.notna(passed_df_processed), None)

        passed_df_filename = f"{secure_filename(selected_collection)}_passed_df.parquet"
        passed_df_filepath = os.path.join(app.config['UPLOAD_FOLDER'], passed_df_filename)
        passed_df_final.to_parquet(passed_df_filepath)
        session['passed_df_path'] = passed_df_filepath

        error_df_filename = f"{secure_filename(selected_collection)}_error_df.parquet"
        error_df_filepath = os.path.join(app.config['UPLOAD_FOLDER'], error_df_filename)
        error_df_from_validation.to_parquet(error_df_filepath) 
        session['error_df_path'] = error_df_filepath
        
        session['unresolved_values_json'] = json.dumps(unresolved_values if unresolved_values else {})

        flash("Data validation and processing complete.", "success")
        return render_template('validate_data.html',
                               title='Step 3: Validate Data',
                               selected_collection=selected_collection,
                               passed_df_count=len(passed_df_final),
                               error_df_count=len(error_df_from_validation),
                               unresolved_values_json=session['unresolved_values_json'])
    except Exception as e:
        flash(f"Error during data validation: {e}", "error")
        app.logger.error(f"Validation error for {selected_collection}: {e}", exc_info=True)
        return redirect(url_for('process_dependencies'))


@app.route('/upload_data_form')
def upload_data_form():
    passed_df_path = session.get('passed_df_path')
    selected_collection = session.get('selected_collection')

    if not passed_df_path or not selected_collection:
        flash("Validated data not found or session expired. Please start from Step 3 (Validate Data).", "error")
        return redirect(url_for('validate_data')) # Or index, depending on desired flow

    if not os.path.exists(passed_df_path):
        flash(f"Passed data file not found at {passed_df_path}. Please re-validate.", "error")
        return redirect(url_for('validate_data'))
    
    try:
        # Optionally, load df to get count if not storing in session
        passed_df = pd.read_parquet(passed_df_path)
        if passed_df.empty:
            flash("No data passed validation. Cannot proceed to upload.", "warning")
            # No need to redirect here, template can handle this message
    except Exception as e:
        flash(f"Error reading passed data file: {e}. Please re-validate.", "error")
        return redirect(url_for('validate_data'))


    db = DatabaseManager()
    try:
        field_defs = get_field_definitions(db, selected_collection)
        collection_columns = list(field_defs.keys())
    except Exception as e:
        flash(f"Error fetching field definitions for collection '{selected_collection}': {e}", "error")
        collection_columns = [] # Proceed with empty columns so form still loads

    return render_template('upload_data_form.html',
                           title='Step 4: Select Upload Options',
                           selected_collection=selected_collection,
                           collection_columns=collection_columns,
                           passed_df_empty=passed_df.empty if 'passed_df' in locals() else True)


@app.route('/execute_upload', methods=['POST'])
def execute_upload():
    passed_df_path = session.get('passed_df_path')
    selected_collection = session.get('selected_collection')

    if not passed_df_path or not selected_collection:
        flash("Session data missing or expired. Please start from Step 1.", "error")
        return redirect(url_for('index'))

    if not os.path.exists(passed_df_path):
        flash(f"Passed data file not found at {passed_df_path}. Please re-validate.", "error")
        return redirect(url_for('validate_data'))

    try:
        passed_df = pd.read_parquet(passed_df_path)
        if passed_df.empty:
            flash("No data available to upload.", "warning")
            return redirect(url_for('upload_status'))

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


        db = DatabaseManager()
        success = upload_data(db, selected_collection, passed_df, upload_mode, pk=pk_column, conflict_col=conflict_column)
        
        if success:
            flash(f"Successfully uploaded {len(passed_df)} rows to '{selected_collection}' using mode '{upload_mode}'.", "success")
        else:
            # upload_data function should raise an exception on failure, which will be caught below.
            # If it returns False without an exception, this path is taken.
            flash(f"Upload to '{selected_collection}' failed. Check logs for details.", "error")

    except Exception as e:
        flash(f"Error during data upload: {e}", "error")
        app.logger.error(f"Upload execution error for {selected_collection}: {e}", exc_info=True)
    
    return redirect(url_for('upload_status'))


@app.route('/upload_status')
def upload_status():
    return render_template('upload_status.html', title='Upload Status')


if __name__ == '__main__':
    app.run(debug=True)

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
