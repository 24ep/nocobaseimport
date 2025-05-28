# NocoBase Data Importer (Flask Version)

This application allows importing data into NocoBase collections via an Excel file,
with steps for dependency processing, validation, and various upload modes. It now also includes features like automatic backup of original uploaded files to a MinIO server, enhanced job progress tracking with downloadable result files, and an improved user interface.

## Docker Deployment

This application can be easily deployed using Docker. The recommended method is using `docker-compose.yml` as described below.

### Prerequisites

*   Docker and Docker Compose installed.
*   (Redis, MinIO, and PostgreSQL services for the importer app and a sample NocoBase target are provided in the `docker-compose.yml`)

### Environment Configuration

1.  Copy the `.env.example` file to `.env`:
    ```bash
    cp .env.example .env
    ```
2.  Edit the `.env` file with your desired configurations:
    *   `FLASK_SECRET_KEY`: A strong, unique secret key for Flask sessions.
    *   `FLASK_UPLOAD_FOLDER`: The folder where uploaded files will be temporarily stored. When using `docker-compose.yml` (recommended), this is set to `/app/flask_nocobase_importer/uploads` inside the container to match the mounted volume.
    *   `FLASK_RUN_PORT`: Port inside the container the app runs on (defaults to 5000, matches Dockerfile).
    *   `REDIS_URL`: The connection URL for your Redis instance (e.g., `redis://redis:6379/0` when using the provided `docker-compose.yml`).
    *   `MINIO_ENDPOINT`: Endpoint of your MinIO server (e.g., `http://minio:9000` when using the provided `docker-compose.yml`).
    *   `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET_NAME`: Credentials and bucket for MinIO.
    *   `MINIO_SECURE`: Set to `True` if your MinIO connection uses HTTPS, `False` otherwise.
    *   `IMPORTER_DB_NAME`, `IMPORTER_DB_USER`, `IMPORTER_DB_PASSWORD`: Credentials for the importer application's own PostgreSQL database.
    *   `SAMPLE_NOCOBASE_DB_NAME`, `SAMPLE_NOCOBASE_DB_USER`, `SAMPLE_NOCOBASE_DB_PASSWORD`: Credentials for the optional sample target NocoBase PostgreSQL database.

### Deployment with Docker Compose

For a comprehensive local development, testing, and production-like environment, Docker Compose is used to manage the web application, RQ worker, Redis, MinIO, and PostgreSQL services.

1.  **Prerequisites**: Ensure you have Docker and Docker Compose installed.
2.  **Environment Configuration**: Create or update your `.env` file from `.env.example` with your configurations.
    *   MinIO connection variables (`MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET_NAME`) should be configured. The `docker-compose.yml` sets up a MinIO service using these for root credentials and configures the web/worker services to connect to it at `http://minio:9000`.
    *   Database credentials for the importer app (`IMPORTER_DB_NAME`, `IMPORTER_DB_USER`, `IMPORTER_DB_PASSWORD`) and the sample NocoBase DB (`SAMPLE_NOCOBASE_DB_NAME`, etc.) are also required.
3.  **Start Services**: Navigate to the project root directory and run:
    ```bash
    docker-compose up -d --build
    ```
    This command will:
    *   Build the Docker image for the `web` and `worker` services (if not already built or if changes are detected).
    *   Start the Flask web server (using Gunicorn), RQ worker, Redis, MinIO, and the importer application's own PostgreSQL database (`importer_db`) in detached mode.
    *   It also starts an optional `sample_nocobase_db` PostgreSQL service. This service can be used as a target database for testing imports. It will be initially empty and accessible to the importer application at host `sample_nocobase_db` on port `5432`.

4.  **Accessing Services**:
    *   Application: The application should be accessible at `http://localhost:5000`.
    *   MinIO Console: The MinIO console will be available at `http://localhost:9001` (using the `MINIO_ACCESS_KEY` and `MINIO_SECRET_KEY` from your `.env` file as `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD` respectively).
    *   Importer App DB: Accessible on host port `5433` (maps to `5432` in container).
    *   Sample NocoBase DB: Accessible on host port `5434` (maps to `5432` in container).
5.  **Persistent Data**: The `docker-compose.yml` uses named volumes to persist data for Redis (`redis_data`), MinIO (`minio_data`), the importer app's database (`importer_db_data`), and the sample NocoBase database (`sample_nocobase_db_data`). Uploaded files are persisted in the `./persistent_uploads` directory on the host, which is mounted into the `web` and `worker` containers.
6.  **Stopping Services**: To stop all services, run from another terminal:
    ```bash
    docker-compose down
    ```

#### Using the Sample NocoBase Target Database (`sample_nocobase_db`)

The `docker-compose.yml` includes an additional PostgreSQL service named `sample_nocobase_db`. This database is provided as a convenient, empty target for testing your import configurations.

To use it:
1.  Ensure the `sample_nocobase_db` service is running (it starts by default with `docker-compose up -d`).
2.  In the NocoBase Importer application, navigate to "Manage NocoBase Profiles" and add a new profile.
3.  Use the following connection details for the profile:
    *   **Profile Name**: A friendly name, e.g., `Local Sample NocoBase DB`
    *   **Host**: `sample_nocobase_db` (this is the service name within the Docker network)
    *   **Port**: `5432` (the internal port for PostgreSQL)
    *   **Database Name**: The value you set for `SAMPLE_NOCOBASE_DB_NAME` in your `.env` file (defaults to `sample_nocobase_data`).
    *   **User**: The value you set for `SAMPLE_NOCOBASE_DB_USER` in your `.env` file (defaults to `sample_user`).
    *   **Password**: The value you set for `SAMPLE_NOCOBASE_DB_PASSWORD` in your `.env` file.
4.  Save the profile. You can now select this profile when starting a new import.

**Note**: This `sample_nocobase_db` is a plain PostgreSQL instance. For it to function as a true NocoBase database, you would need to initialize it with the NocoBase schema separately. This setup does not do that for you; it merely provides the PostgreSQL instance.

### Manual Docker Build and Run (Advanced)

While `docker-compose` (described above) is the recommended method for running the application and all its services, you can also build and run the web container manually.

#### Building the Docker Image

Navigate to the project root directory (where the `Dockerfile` is located) and run:

```bash
docker build -t nocobase-importer .
```

#### Running the Docker Container (Web only)

To run the container (assuming Redis, MinIO, and PostgreSQL are accessible):

```bash
docker run -d \
  -p 5000:5000 \
  --env-file .env \
  -v "$(pwd)/persistent_uploads":/app/flask_nocobase_importer/uploads \
  --name nocobase-importer-app \
  nocobase-importer
```
Note: You'll need to ensure all environment variables in `.env` (including for Redis, MinIO, DBs) are correctly pointing to your externally managed services.

### Deployment with Portainer (using Stacks)

The recommended way to deploy this application using Portainer is via its "Stacks" feature, which utilizes the `docker-compose.yml` file.

1.  **Prepare your Environment File**:
    *   Create a `.env` file on your system based on `.env.example`.
    *   Fill in all necessary values, especially for `FLASK_SECRET_KEY`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `IMPORTER_DB_PASSWORD`, `SAMPLE_NOCOBASE_DB_PASSWORD`, etc.

2.  **Create a New Stack in Portainer**:
    *   Navigate to "Stacks" in Portainer and click "Add stack".
    *   **Name**: Give your stack a name (e.g., `nocobase-importer-stack`).
    *   **Build method**: Choose "Web editor".
    *   **Web editor**: Copy the entire content of the `docker-compose.yml` file from this project and paste it into the editor.
    *   **Environment variables**:
        *   Portainer Stacks can utilize `.env` files in different ways depending on the Portainer version and setup.
        *   Focus on defining runtime secrets in the "Environment variables" section if not using a `.env` file effectively. The `docker-compose.yml` references these from the environment it runs in (e.g., `FLASK_SECRET_KEY`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `IMPORTER_DB_NAME`, `IMPORTER_DB_USER`, `IMPORTER_DB_PASSWORD`, `SAMPLE_NOCOBASE_DB_NAME`, `SAMPLE_NOCOBASE_DB_USER`, `SAMPLE_NOCOBASE_DB_PASSWORD`).
        *   Service-to-service connection details like `DB_HOST=importer_db` or `MINIO_ENDPOINT=http://minio:9000` are usually hardcoded in the `docker-compose.yml` or in the application's specific environment variables pointing to service names.

3.  **Deploy the Stack**: Click "Deploy the stack". Portainer will pull the necessary images and start all services defined in the `docker-compose.yml`.

4.  **Accessing Services**:
    *   Application: `http://<your-portainer-host-ip>:5000`
    *   MinIO Console: `http://<your-portainer-host-ip>:9001`
    *   Importer App DB: `your-portainer-host-ip:5433`
    *   Sample NocoBase DB: `your-portainer-host-ip:5434`

5.  **Volumes**: Portainer will manage the volumes defined in `docker-compose.yml`. The `./persistent_uploads` directory specified in the `volumes` section for `web` and `worker` services will be created on the Docker host where Portainer executes the stack.

### Running RQ Workers Manually (for Local Development without Docker Compose)

While Docker Compose (`docker-compose up`) is the recommended way to run all services including the RQ worker during development, you can also run the RQ worker manually in your local environment. This requires careful setup:

1.  **Activate Virtual Environment**: Ensure you have activated the Python virtual environment where all dependencies from `requirements.txt` are installed.
    ```bash
    # Example:
    # source venv/bin/activate
    ```

2.  **Set FLASK_APP Environment Variable**: The Flask CLI needs to know where your application instance is. Set this variable in your terminal session before running the worker.
    *   For Linux/macOS (bash/zsh):
        ```bash
        export FLASK_APP=flask_nocobase_importer.app
        ```
    *   For Windows (Command Prompt):
        ```bash
        set FLASK_APP=flask_nocobase_importer.app
        ```
    *   For Windows (PowerShell):
        ```powershell
        $env:FLASK_APP="flask_nocobase_importer.app"
        ```
    You must set this from the project root directory (the directory containing the `flask_nocobase_importer` folder and your `.env` file).

3.  **Ensure `.env` File is Present**: The worker tasks may need database credentials or other configurations defined in your `.env` file. Make sure your `.env` file (copied from `.env.example` and configured) is present in the current working directory (project root) when you start the worker. This includes `IMPORTER_DB_*` variables, `REDIS_URL`, and `MINIO_*` variables.

4.  **Ensure Redis, MinIO, and Importer DB are Running**: The RQ worker needs to connect to a Redis server. The tasks may also interact with MinIO and the importer application's database. Make sure these services are running and accessible at the URLs specified in your environment variables.

5.  **Start the Worker**: From your project root directory, run the following command:
    ```bash
    python -m flask rq worker -u ${REDIS_URL:-redis://localhost:6379/0} default
    ```
    Or, if `FLASK_APP` is correctly set and your environment resolves the `flask` command directly:
    ```bash
    flask rq worker -u ${REDIS_URL:-redis://localhost:6379/0} default
    ```
    This command tells the worker to connect to your Redis instance (using the `REDIS_URL` from your environment or defaulting to `redis://localhost:6379/0`) and process jobs on the `default` queue. You can list multiple queues if needed.

**Troubleshooting Worker Startup:**
*   **`Error: Could not locate a Flask application.`**: This usually means `FLASK_APP` is not set correctly or you are not in the project root directory. Verify the variable and your current path.
*   **`Error: No such command 'rq'.`**: This typically means the virtual environment isn't activated, `Flask-RQ2` is not installed properly, or there's an issue with your Python environment's PATH. Ensure `pip install -r requirements.txt` was successful.
*   **Connection Errors to Redis/MinIO/DB**: Ensure these services are running and accessible, and that your `.env` file has the correct connection details.

Using `python -m flask ...` can sometimes be more reliable than just `flask ...` if you have multiple Python versions or complex environments.

## Security Considerations

It's crucial to deploy and manage this application with security in mind. Here are some important considerations:

### HTTPS Enforcement
The application itself does not handle SSL/TLS termination. For production deployments, **always run this application behind a reverse proxy** (e.g., Nginx, Traefik, Caddy) that is configured to handle HTTPS and terminate SSL/TLS. Use tools like Let's Encrypt for free SSL certificates. Enforcing HTTPS is critical to protect data in transit.

### Secret Management
*   **`FLASK_SECRET_KEY`**: This key is used to sign session cookies and other security-sensitive tokens. The default value in `.env.example` **must be changed** to a strong, unique, and random string for your production environment. Keep this key confidential.
*   **Database Credentials & MinIO Keys**: Ensure your `.env` file containing database credentials (for the importer app DB, sample NocoBase DB) and MinIO keys (`MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`) is kept secure, has appropriate file permissions, and is **never committed to version control**.
*   **Environment Variables**: In general, prefer environment variables for all secrets, as demonstrated in the Docker deployment setup.

### File Uploads
*   **Size Limit**: The application is configured with a maximum file upload size (currently 16MB, set by `MAX_CONTENT_LENGTH` in `app.py`) to prevent denial-of-service attacks via excessively large uploads.
*   **Filename Sanitization**: Uploaded filenames are processed using Werkzeug's `secure_filename` to prevent directory traversal attacks or malicious filenames. MinIO destination filenames are also sanitized in the tasks.
*   **Content Type**: While the application primarily expects Excel files, further validation of file content or MIME types could be added if stricter controls are needed beyond what `pandas.read_excel` provides.

### Dependency Management
*   Regularly check your application's dependencies (listed in `requirements.txt`) for known vulnerabilities.
*   Tools like `pip-audit` (e.g., `pip-audit requirements.txt`) or services like GitHub's Dependabot can help automate this process. Keep dependencies updated to their latest secure versions.

### Session Cookies
The application is configured to use secure session cookie attributes:
*   `SESSION_COOKIE_SECURE=True` (when `FLASK_SESSION_COOKIE_SECURE=True` in env, for HTTPS): Ensures cookies are only sent over HTTPS.
*   `SESSION_COOKIE_HTTPONLY=True`: Prevents client-side JavaScript from accessing the session cookie, mitigating XSS risks.
*   `SESSION_COOKIE_SAMESITE='Lax'`: Provides protection against Cross-Site Request Forgery (CSRF) for GET requests.

### HTTP Security Headers
*   The application uses `Flask-Talisman` to set various important HTTP security headers, such as:
    *   `X-Frame-Options` (to prevent clickjacking)
    *   `X-Content-Type-Options` (to prevent MIME-sniffing)
    *   `Strict-Transport-Security (HSTS)` (if configured, typically by Talisman based on HTTPS detection)
    *   `Content-Security-Policy (CSP)`
*   **Content Security Policy (CSP)**: Flask-Talisman applies a default CSP (e.g., `default-src 'self'; object-src 'none';`). This is a strong baseline. If your application needs to load resources from external domains (e.g., CDNs for CSS/JS, external images, APIs), you may need to customize the CSP policy in `app.py` when initializing Talisman.

### Running in Docker
*   The provided `Dockerfile` and `docker-compose.yml` aim for a secure setup. Ensure your Docker host is secured and that you manage your `.env` files and any mapped volumes appropriately.
```
