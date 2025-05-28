import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import json
# from io import BytesIO, StringIO # StringIO not used, BytesIO can be used for file uploads in Flask later
import time
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import Tuple, Optional
import logging # Added for logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Assuming DatabaseManager will be imported from db_manager in the Flask app
# from .db_manager import DatabaseManager

# ---------------------------
# Data Processing Functions
# ---------------------------

def parse_options(options_value):
    if isinstance(options_value, dict):
        return options_value
    try:
        return json.loads(options_value)
    except Exception:
        return {}

def get_collections(db: 'DatabaseManager'):
    query = "SELECT name FROM public.collections"
    rows = db.fetch_all(query)
    return [row[0] for row in rows]

def get_field_definitions(db: 'DatabaseManager', collection):
    query = "SELECT name, type, options FROM public.fields WHERE collection_name = %s"
    rows = db.fetch_all(query, (collection,))
    field_defs = {}
    for field_name, field_type, field_options in rows:
        opts = parse_options(field_options)
        ui_schema = opts.get("uiSchema", {})
        component_props = ui_schema.get("x-component-props", {})
        allowed_options = []
        if "enum" in ui_schema:
            allowed_options = [str(opt["value"]) if isinstance(opt, dict) else str(opt) 
                               for opt in ui_schema["enum"]]
        elif "options" in component_props:
            allowed_options = [str(opt["value"]) if isinstance(opt, dict) else str(opt)
                               for opt in component_props.get("options", [])]

        field_defs[field_name] = {
            "type": field_type,
            "options": opts,
            "multiple": component_props.get("multiple", False),
            "allowed_options": allowed_options,
            "is_nullable": True
        }

    column_query = """
    SELECT column_name, data_type, character_maximum_length, 
           is_nullable = 'YES' as is_nullable
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = %s
    """
    column_rows = db.fetch_all(column_query, (collection,))
    for col in column_rows:
        col_name = col[0]
        if col_name in field_defs:
            field_defs[col_name].update({
                "data_type": col[1],
                "char_max_len": col[2],
                "is_nullable": col[3]
            })
    return field_defs


def resolve_extra_columns(df, field_defs):
    """
    Identifies extra columns in the DataFrame compared to field_defs.
    Logs extra columns and returns a DataFrame with only known columns.
    """
    known_cols = [col for col in df.columns if col in field_defs]
    extra_cols = [col for col in df.columns if col not in field_defs]
    
    if extra_cols:
        logging.warning(f"Extra columns found and will be dropped: {extra_cols}")
    
    return df[known_cols]

def get_allowed_options(db: 'DatabaseManager', target, targetKey):
    if not target or not targetKey:
        return []
    query = sql.SQL("SELECT DISTINCT {} FROM public.{}").format(
        sql.Identifier(targetKey), sql.Identifier(target)
    )
    results = db.fetch_all(query)
    return [str(r[0]) for r in results if r[0] is not None]

def get_candidate_unique_key(db: 'DatabaseManager', target):
    query = """
    SELECT kcu.column_name
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu
      ON tc.constraint_name = kcu.constraint_name
     AND tc.table_schema = kcu.table_schema
    WHERE tc.table_schema = 'public'
      AND tc.table_name = %s
      AND tc.constraint_type = 'UNIQUE'
      AND kcu.column_name <> 'id'
    LIMIT 1;
    """
    result = db.execute(query, (target,), fetchone=True)
    if result:
        return result[0]
    query2 = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = %s
      AND column_name <> 'id'
      AND data_type IN ('character varying', 'text')
    LIMIT 1;
    """
    result = db.execute(query2, (target,), fetchone=True)
    return result[0] if result else None

def preprocess_foreign_keys(df, field_defs, db: 'DatabaseManager'):
    total = len(df)
    start_time = time.time()
    lookup_cache = {}
    logging.info("Starting foreign key preprocessing...")

    for col, spec in field_defs.items():
        opts = spec.get("options", {})
        if "target" in opts and "targetKey" in opts and col in df.columns:
            target = opts.get("target")
            targetKey = opts.get("targetKey")
            unique_key = get_candidate_unique_key(db, target) or targetKey

            if target not in lookup_cache:
                try:
                    query = sql.SQL("SELECT {}, id FROM public.{}").format(
                        sql.Identifier(unique_key), sql.Identifier(target)
                    )
                    results = db.fetch_all(query)
                    mapping = {str(row[0]).strip().lower(): str(row[1]) for row in results if row[0] is not None}
                    lookup_cache[target] = (unique_key, mapping)
                except Exception as e:
                    logging.error(f"Error preloading lookup for target '{target}': {str(e)}")
                    # Depending on strictness, could raise error or continue
                    # raise  # Or continue if this target is not critical
                    continue # Continue for now

    for i, idx in enumerate(df.index):
        for col, spec in field_defs.items():
            opts = spec.get("options", {})
            if "target" in opts and "targetKey" in opts and col in df.columns:
                target = opts.get("target")
                if target not in lookup_cache:
                    continue

                unique_key, mapping = lookup_cache[target]
                value = df.at[idx, col]
                if pd.isna(value) or value is None:
                    continue

                field_type = spec.get("type")
                if isinstance(value, str) and "|" in value:
                    parts = [v.strip() for v in value.split("|")]
                    new_ids = []
                    for part in parts:
                        key = part.lower()
                        if key in mapping:
                            new_ids.append(mapping[key])
                    if new_ids:
                        if field_type == "belongsToArray":
                            df.at[idx, f"{col}_id"] = f"{{{','.join(new_ids)}}}"
                        else:
                            df.at[idx, f"{col}_id"] = new_ids[0]
                else:
                    key = str(value).strip().lower()
                    if key in mapping:
                        if field_type == "belongsToArray":
                            df.at[idx, f"{col}_id"] = f"{{{mapping[key]}}}"
                        else:
                            df.at[idx, f"{col}_id"] = mapping[key]
        
        if (i % 100 == 0 or i == total - 1) and total > 0: # Log progress periodically
            progress = int(((i + 1) / total) * 100)
            elapsed = time.time() - start_time
            remaining_seconds = (elapsed / (i + 1)) * (total - (i + 1)) if (i+1) > 0 else 0
            remaining_formatted = datetime.utcfromtimestamp(remaining_seconds).strftime('%H:%M:%S')
            logging.info(f"Preprocessing FKs: {progress}% | Processed {i+1}/{total} rows. Estimated time remaining: {remaining_formatted}")
    
    logging.info("Foreign key preprocessing finished.")
    return df

def clean_dataframe(df):
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    return df.where(pd.notna(df), None)

def validate_dataframe(df, field_defs, db: 'DatabaseManager'):
    unresolved_values = defaultdict(lambda: defaultdict(int))
    fk_cache = {}

    for field, spec in field_defs.items():
        if spec['type'] in ['belongsTo', 'belongsToArray']:
            target = spec['options'].get('target')
            unique_key = get_candidate_unique_key(db, target)
            if not unique_key:
                unique_key = spec['options'].get('targetKey', 'id')
            if target not in fk_cache:
                fk_cache[target] = {
                    'target_key': unique_key,
                    'values': set(x.lower() for x in get_allowed_options(db, target, unique_key))
                }

    df = df.copy()
    df['errors'] = [[] for _ in range(len(df))]

    for col in field_defs:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].where(df[col].isnull(), df[col].str.strip())
            df[col] = df[col].where(pd.notna(df[col]), None)

    for col, spec in field_defs.items():
        if col not in df.columns:
            continue
        if not spec.get('is_nullable', True):
            mask = df[col].isnull()
            error_msg = f"Required field {col} is missing"
            for idx in df[mask].index:
                df.at[idx, 'errors'].append(error_msg)

    for col, spec in field_defs.items():
        if col not in df.columns:
            continue
        max_length = spec.get('char_max_len', 255) 
        if max_length is None: 
            max_length = 255
        mask = df[col].notnull()
        for idx, value in df.loc[mask, col].items():
            if len(str(value)) > max_length:
                error_msg = f"Field {col} exceeds maximum length of {max_length}"
                df.at[idx, 'errors'].append(error_msg)

    numeric_types = {"double": float, "float": float, "number": float, "integer": int}
    for col, spec in field_defs.items():
        if col not in df.columns:
            continue
        conversion_func = numeric_types.get(spec['type'])
        if conversion_func is not None:
            mask = df[col].notnull()
            for idx, value in df.loc[mask, col].items():
                try:
                    conversion_func(value)
                except (ValueError, TypeError):
                    error_msg = f"Invalid type for field {col}: expected {spec['type']} but got '{value}'"
                    df.at[idx, 'errors'].append(error_msg)

    for col, spec in field_defs.items():
        if col not in df.columns:
            continue
        if spec['type'] in ['belongsTo', 'belongsToArray']:
            target = spec['options'].get('target')
            allowed_set = fk_cache.get(target, {}).get('values', set())
            if spec['type'] == 'belongsTo':
                mask = df[col].notnull()
                cleaned = df.loc[mask, col].astype(str).str.strip()
                cleaned_lower = cleaned.str.lower()
                invalid_mask = ~cleaned_lower.isin(allowed_set)
                for idx in cleaned[invalid_mask].index:
                    v = cleaned.at[idx]
                    unresolved_values[col][v] = 1 # Changed from += 1
                    error_msg = f"Invalid references in {col}: {{{v}}} not found in {target}"
                    df.at[idx, 'errors'].append(error_msg)
            else: 
                mask = df[col].notnull()
                for idx in df.loc[mask].index:
                    value_str = str(df.at[idx, col])
                    values = [v.strip() for v in value_str.split('|')]
                    missing = [v for v in values if v.lower() not in allowed_set]
                    if missing:
                        for m in missing:
                            unresolved_values[col][m] = 1 # Changed from += 1
                        error_msg = f"Invalid references in {col}: {set(missing)} not found in {target}"
                        df.at[idx, 'errors'].append(error_msg)

    passed_df = df[df['errors'].apply(len) == 0].drop(columns=['errors'])
    error_df = df[df['errors'].apply(len) > 0]
    return passed_df, error_df, unresolved_values


def read_excel_file(uploaded_file_path, selected_sheet_name=None):
    start_time = time.time()
    try:
        xls = pd.ExcelFile(uploaded_file_path)
    except FileNotFoundError:
        logging.error(f"Excel file not found at path: {uploaded_file_path}")
        raise
    except Exception as e:
        logging.error(f"Error opening Excel file {uploaded_file_path}: {e}")
        raise

    sheet_names = xls.sheet_names
    
    if selected_sheet_name:
        if selected_sheet_name not in sheet_names:
            logging.error(f"Sheet name '{selected_sheet_name}' not found in Excel file. Available sheets: {sheet_names}")
            raise ValueError(f"Sheet name '{selected_sheet_name}' not found.")
        selected_sheet = selected_sheet_name
    elif sheet_names:
        selected_sheet = sheet_names[0]
        if len(sheet_names) > 1:
            logging.info(f"Multiple sheets found: {sheet_names}. Auto-selected: {selected_sheet}.")
    else:
        logging.error("No sheets found in the Excel file.")
        raise ValueError("No sheets found in Excel file.")

    skip_rows = range(1, 5) if selected_sheet.strip().lower() == "visitor_information" else None
    
    logging.info(f"Loading Excel data from sheet: '{selected_sheet}'...")
    df = pd.read_excel(
        xls, 
        sheet_name=selected_sheet,
        dtype=str,
        engine="openpyxl",
        skiprows=skip_rows
    )
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df = df.where(pd.notna(df), None)
    df = df.replace({"nan": None, np.nan: None}) # Replace with None, not empty string
    logging.info(f"Loaded {len(df)} rows in {time.time()-start_time:.2f}s.")
    return df


def preprocess_array_fields(df, field_defs):
    for col, spec in field_defs.items():
        if spec.get("type") == "array" and col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.dumps([i.strip() for i in x.split('|')])
                if isinstance(x, str) and "|" in x else (
                    json.dumps([x.strip()]) if isinstance(x, str) else (
                        x if pd.isna(x) else json.dumps([str(x).strip()]) 
                    )
                )
            )
    return df


def extract_and_rename(df, columns_to_extract, rename_map):
    existing_cols = [col for col in columns_to_extract if col in df.columns]
    missing_cols = [col for col in columns_to_extract if col not in df.columns]
    
    if missing_cols:
        logging.warning(f"Columns to extract but not found in DataFrame: {missing_cols}. They will be added as NA.")

    new_df = df[existing_cols].copy()
    
    for col in missing_cols:
        new_df[col] = pd.NA 
    
    new_df.rename(columns=rename_map, inplace=True)
    return new_df


def process_calling_list(df):
    phone_columns = [
        ("mobile_no_1", "number_with_extension"),
        ("mobile_no_2", "number_with_extension"),
        ("telephone_extension_1", "number_with_extension"),
        ("telephone_extension_2", "number_with_extension")
    ]
    
    phone_dfs = []
    for src_col, target_col in phone_columns:
        if src_col in df.columns:
            df_phone = extract_and_rename(
                df, 
                [src_col], 
                {src_col: target_col}
            )
            df_phone['source_type'] = src_col
            phone_dfs.append(df_phone)
    
    if not phone_dfs: 
        logging.info("No source phone columns found for processing calling list.")
        return pd.DataFrame(columns=['number_with_extension', 'source_type'])

    calling_list = pd.concat(phone_dfs, ignore_index=True)
    
    calling_list = (
        calling_list
        .dropna(subset=['number_with_extension'])
        .drop_duplicates(subset=['number_with_extension'])
        .pipe(validate_phone_numbers)
        .reset_index(drop=True)
    )
    return calling_list

def validate_phone_numbers(df):
    df['number_with_extension'] = (
        df['number_with_extension']
        .astype(str) 
        .str.replace(r'\D', '', regex=True)
        .str.replace(r'^0', '', regex=True)
        .apply(lambda x: x if len(x) >= 8 else None)
    )
    return df.dropna(subset=['number_with_extension'])

def update_batches(cursor, table, df, pk, batch_size=10000):
    # This function seems robust, ensure exceptions are handled by the caller (upload_data)
    table_name = sql.Identifier(table)
    pk_name = sql.Identifier(pk)
    update_columns = [col for col in df.columns if col not in [pk, 'updated_at', 'created_at']]
    if not update_columns:
        raise ValueError("No updatable columns found in DataFrame for update_batches.")
    set_elements = [sql.SQL("{} = %s").format(sql.Identifier(col)) for col in update_columns]
    set_elements.append(sql.SQL("updated_at = CURRENT_TIMESTAMP"))
    set_clause = sql.SQL(", ").join(set_elements)
    query = sql.SQL("UPDATE {table} SET {set_clause} WHERE {pk} = %s").format(
        table=table_name, set_clause=set_clause, pk=pk_name
    )
    data_cols = update_columns + [pk]
    data = df[data_cols].where(pd.notnull(df), None).to_numpy().tolist()
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        cursor.executemany(query, batch)
    return True


def upload_batches(cursor, table, df, batch_size=10000):
    # This function seems robust, ensure exceptions are handled by the caller (upload_data)
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        columns = [col for col in batch.columns if col != 'created_at']
        if not columns: # Avoid inserting if only 'created_at' or no columns
            logging.warning(f"Skipping batch for table {table} as no valid columns found for insert.")
            continue
        query_template = sql.SQL("INSERT INTO public.{table} ({cols}, created_at) VALUES ({placeholders}, CURRENT_TIMESTAMP)").format(
            table=sql.Identifier(table),
            cols=sql.SQL(', ').join(map(sql.Identifier, columns)),
            placeholders=sql.SQL(', ').join(sql.Placeholder() * len(columns))
        )
        records = [tuple(x) for x in batch[columns].to_numpy()]
        if records:
             cursor.executemany(query_template, records)


def upsert_batches(cursor, table, df, conflict_col, batch_size=10000):
    # This function seems robust, ensure exceptions are handled by the caller (upload_data)
    if conflict_col not in df.columns:
        raise ValueError(f"Conflict column '{conflict_col}' not found in DataFrame for upsert_batches.")
    insert_columns = [col for col in df.columns if col != 'created_at']
    if not insert_columns:
        logging.warning(f"Skipping upsert for table {table} as no valid columns found for insert.")
        return True

    quoted_insert_cols = [sql.Identifier(col) for col in insert_columns]
    update_cols = [col for col in insert_columns if col not in [conflict_col, 'updated_at']]
    set_elements = [sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(col), sql.Identifier(col)) for col in update_cols]
    set_elements.append(sql.SQL("updated_at = CURRENT_TIMESTAMP"))
    set_clause = sql.SQL(", ").join(set_elements)
    query = sql.SQL("INSERT INTO {table} ({cols}, created_at) VALUES ({vals}, CURRENT_TIMESTAMP) ON CONFLICT ({conflict}) DO UPDATE SET {set_clause}").format(
        table=sql.Identifier(table),
        cols=sql.SQL(', ').join(quoted_insert_cols),
        vals=sql.SQL(', ').join([sql.Placeholder()] * len(insert_columns)),
        conflict=sql.Identifier(conflict_col),
        set_clause=set_clause
    )
    data_df = df[insert_columns].where(pd.notnull(df[insert_columns]), None)
    data = [tuple(row) for row in data_df.to_numpy()]
    if not data: 
        return True
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        cursor.executemany(query, batch)
    return True

def upload_data(db: 'DatabaseManager', collection, df, mode="insert", pk=None, conflict_col=None):
    conn = None 
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        if mode == "insert":
            upload_batches(cursor, collection, df)
        elif mode == "update":
            if not pk:
                logging.error("Primary key (pk) must be provided for update mode.")
                raise ValueError("Primary key (pk) must be provided for update mode.")
            update_batches(cursor, collection, df, pk)
        elif mode == "insert on duplicate update":
            if not conflict_col:
                logging.error("Conflict column must be provided for 'insert on duplicate update' mode.")
                raise ValueError("Conflict column must be provided for 'insert on duplicate update' mode.")
            upsert_batches(cursor, collection, df, conflict_col)
        else:
            logging.error(f"Invalid upload mode: {mode}")
            raise ValueError(f"Invalid upload mode: {mode}")
        conn.commit()
        logging.info(f"Data upload successful for collection '{collection}', mode '{mode}'.")
        return True
    except Exception as e:
        if conn: 
            conn.rollback()
        logging.error(f"Upload error for collection '{collection}', mode '{mode}': {str(e)}")
        # Re-raise the exception so the caller in app.py can handle it (e.g., show flash message)
        raise  
    finally:
        if conn: 
            cursor.close()
            conn.close()


def get_unique_constraint_name(db: 'DatabaseManager', table: str, column: str) -> Optional[str]:
    query = """
    SELECT tc.constraint_name
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu
      ON tc.constraint_name = kcu.constraint_name
     AND tc.table_schema = kcu.table_schema
    WHERE tc.table_schema = 'public'
      AND tc.table_name   = %s
      AND tc.constraint_type IN ('UNIQUE') 
      AND kcu.column_name = %s
    LIMIT 1;
    """
    rows = db.fetch_all(query, (table, column))
    return rows[0][0] if rows else None

def _do_upsert(cursor, table_name: str, df: pd.DataFrame, conflict_col: str) -> int:
    if conflict_col in df.columns:
        df = df.drop_duplicates(subset=[conflict_col], keep="last")
    if df.empty:
        logging.info(f"DataFrame is empty after deduplication on {conflict_col} for table {table_name}, skipping upsert.")
        return 0

    cols = list(df.columns)
    col_idents  = [sql.Identifier(c) for c in cols]
    update_cols = [c for c in cols if c != conflict_col] # These are columns to be updated if conflict occurs
    
    if update_cols: # If there are columns to update (other than the conflict column)
        set_clause = sql.SQL(", ").join(
            sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(c), sql.Identifier(c))
            for c in update_cols 
        )
        conflict_clause = sql.SQL(" ON CONFLICT ({}) DO UPDATE SET {}").format(
            sql.Identifier(conflict_col),
            set_clause
        )
    else: # If only the conflict column is present, or no other columns are updatable
        conflict_clause = sql.SQL(" ON CONFLICT ({}) DO NOTHING").format(
            sql.Identifier(conflict_col)
        )

    insert_sql = sql.SQL("INSERT INTO public.{} ({}) VALUES %s{}").format(
        sql.Identifier(table_name),
        sql.SQL(", ").join(col_idents),
        conflict_clause
    )

    data = [tuple(r) for r in df.itertuples(index=False, name=None)]
    if not data:
        return 0

    execute_values(cursor, insert_sql, data) # page_size removed
    return cursor.rowcount

def process_dependency(
    db: 'DatabaseManager',
    dep_name: str,
    config: dict,
    main_df: pd.DataFrame
) -> Tuple[bool, Optional[pd.DataFrame], int]:
    error_df_validation = None # To store errors from validate_dataframe
    rowcount = 0
    try:
        df_to_process = main_df
        if df_to_process.empty:
            logging.info(f"No input rows for dependency '{dep_name}', skipping.")
            return True, None, 0
        
        logging.info(f"Processing dependency: {dep_name} - before clean data")
        dep_df = extract_and_rename(df_to_process, config["cols"], config["mapping"])
        dep_df = clean_dataframe(dep_df)
        logging.info(f"Processing dependency: {dep_name} - after clean data")

        field_defs = get_field_definitions(db, dep_name)
        passed_df, error_df_validation, _ = validate_dataframe(dep_df, field_defs, db)
        
        # Pass db to preprocess_foreign_keys, logging will be handled inside
        passed_df = preprocess_foreign_keys(passed_df, field_defs, db)
        passed_df = preprocess_array_fields(passed_df, field_defs)
        passed_df = drop_relationship_fields(passed_df, field_defs)
        # Ensure None, not NA for database operations. clean_dataframe should handle this.
        passed_df = passed_df.where(~passed_df.isna(), None) 
        logging.info(f"Processing dependency: {dep_name} - after data transformations")
        
        if passed_df.empty:
            logging.warning(f"No valid rows for dependency '{dep_name}' after validation and processing, skipping upsert.")
            # Return validation errors if any, even if no data to upsert
            return True, error_df_validation if not error_df_validation.empty else None, 0

        passed_df = passed_df.drop_duplicates(keep="last")
        rows_to_upsert = len(passed_df)
        logging.info(f"Dependency '{dep_name}': upserting {rows_to_upsert:,} rows")

        if not passed_df.columns.empty:
            conflict_col = passed_df.columns[0]
        else:
            logging.warning(f"Passed_df for dependency '{dep_name}' has no columns, skipping upsert.")
            return True, error_df_validation if not error_df_validation.empty else None, 0


        conn = db.get_connection()
        cur  = conn.cursor()
        try:
            rowcount = _do_upsert(cur, dep_name, passed_df, conflict_col)
            conn.commit()
            logging.info(f"Successfully upserted {rowcount} rows for dependency '{dep_name}'.")
        except Exception as e:
            conn.rollback()
            logging.error(f"Upsert failed for dependency '{dep_name}': {e}")
            # We return the validation error_df, as that's more relevant to the user for fixing input data
            # The upsert error itself is logged.
            return False, error_df_validation if error_df_validation is not None and not error_df_validation.empty else None, 0
        finally:
            cur.close()
            conn.close()
        
        # If upsert was successful, return True, and None for error_df (aligning with Streamlit)
        return True, None, rowcount

    except Exception as e:
        logging.error(f"Error processing dependency '{dep_name}': {e}")
        # Return the validation error df if available, otherwise an empty one or None
        return False, error_df_validation if error_df_validation is not None and not error_df_validation.empty else None, rowcount


def make_position_at_work(row):
    def split_or_default(val, default):
        if pd.isna(val) or not str(val).strip():
            return [default]
        return [v.strip() for v in str(val).split('|')]

    positions = split_or_default(row.get('lookup_position'), 'Unknown position')
    departments = split_or_default(row.get('lookup_department'), '') 
    companies = split_or_default(row.get('lookup_company'), 'Unknown Company')

    max_len = max(len(positions), len(departments), len(companies))
    
    positions.extend([positions[-1] if positions else 'Unknown position'] * (max_len - len(positions)))
    departments.extend([departments[-1] if departments else ''] * (max_len - len(departments)))
    companies.extend([companies[-1] if companies else 'Unknown Company'] * (max_len - len(companies)))


    results = []
    for pos, dept, comp in zip(positions, departments, companies):
        dept_part = f",{dept}" if dept else ''
        results.append(f"{pos}{dept_part} at {comp}")

    return '|'.join(results)

def make_extension(row, num_col, ext_col):
    num = row.get(num_col)
    ext = row.get(ext_col)

    num = None if (pd.isna(num) or str(num).strip() == "") else str(num).strip()
    ext = None if (pd.isna(ext) or str(ext).strip() == "") else str(ext).strip()

    if num and ext:
        return f"{num}-{ext}"
    elif num:
        return num # num is already a string here or None
    elif ext: 
        return ext # Align with Streamlit: return just the extension string
    else:
        return None

# Functions get_fk_options and get_fk_options_with_mapping seem fine, no st calls.
# They are mostly DB interaction and data shaping.

def get_fk_options(db: 'DatabaseManager', target, target_key, label_field=None):
    if not target or not target_key:
        return []
    
    if label_field:
        query = sql.SQL("SELECT DISTINCT {}, {} FROM public.{}").format(
            sql.Identifier(target_key),
            sql.Identifier(label_field),
            sql.Identifier(target)
        )
        results = db.fetch_all(query)
        return [
            {"value": str(row[0]).strip(), "label": str(row[1]).strip()}
            for row in results if row[0] is not None and row[1] is not None
        ]
    else:
        query = sql.SQL("SELECT DISTINCT {} FROM public.{}").format(
            sql.Identifier(target_key),
            sql.Identifier(target)
        )
        results = db.fetch_all(query)
        return [
            {"value": str(row[0]).strip(), "label": str(row[0]).strip()}
            for row in results if row[0] is not None
        ]

def get_fk_options_with_mapping(db: 'DatabaseManager', target, target_key, label_field=None):
    if (not label_field) or (label_field.lower() == "id"):
        candidate = get_candidate_unique_key(db, target)
        if candidate and candidate.lower() != "id":
            label_field = candidate

    if label_field and label_field.lower() != "id":
        query = sql.SQL("SELECT id, {} FROM public.{}").format(
            sql.Identifier(label_field),
            sql.Identifier(target)
        )
        results = db.fetch_all(query)
        mapping = {str(row[1]).strip().lower(): str(row[0]).strip() 
                   for row in results if row[0] is not None and row[1] is not None}
        options = list(mapping.keys())
        return options, mapping
    else:
        query = sql.SQL("SELECT id FROM public.{}").format(sql.Identifier(target))
        results = db.fetch_all(query)
        mapping = {str(row[0]).strip().lower(): str(row[0]).strip() 
                   for row in results if row[0] is not None}
        options = list(mapping.keys())
        return options, mapping

def drop_relationship_fields(df, field_defs): # Added this function as it was in main.py but not data_processing
    for field, spec in field_defs.items():
        if spec.get("type") in ["belongsTo", "belongsToArray"]:
            if field in df.columns:
                df.drop(columns=[field], inplace=True)
    return df
