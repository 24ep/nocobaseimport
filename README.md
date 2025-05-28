# NocoBase Data Importer (Flask Version)

This application allows importing data into NocoBase collections via an Excel file,
with steps for dependency processing, validation, and various upload modes.

## Docker Deployment

This application can be easily deployed using Docker.

### Prerequisites

*   Docker installed.
*   An instance of PostgreSQL database accessible to the container.
*   An instance of Redis accessible to the application and RQ workers.

### Environment Configuration

1.  Copy the `.env.example` file to `.env`:
    ```bash
    cp .env.example .env
    ```
2.  Edit the `.env` file with your actual database credentials and desired Flask configurations:
    *   `DB_HOST`: Your database host.
    *   `DB_PORT`: Your database port.
    *   `DB_NAME`: The name of your NocoBase database.
    *   `DB_USER`: Database user.
    *   `DB_PASSWORD`: Database password.
    *   `FLASK_SECRET_KEY`: A strong, unique secret key for Flask sessions.
    *   `FLASK_UPLOAD_FOLDER`: The folder where uploaded files will be temporarily stored *inside the container*. Defaults to `uploads/` relative to the app. If mapping a volume, ensure this path matches.
    *   `FLASK_RUN_PORT`: Port inside the container the app runs on (defaults to 5000, matches Dockerfile).
    *   `REDIS_URL`: The connection URL for your Redis instance (e.g., `redis://localhost:6379/0`).

### Building the Docker Image

Navigate to the project root directory (where the `Dockerfile` is located) and run:

```bash
docker build -t nocobase-importer .
```

### Running the Docker Container

To run the container:

```bash
docker run -d \
  -p 5000:5000 \
  --env-file .env \
  -v "$(pwd)/persistent_uploads":/app/flask_nocobase_importer/uploads \
  --name nocobase-importer-app \
  nocobase-importer
```

Explanation of flags:
*   `-d`: Run the container in detached mode (in the background).
*   `-p 5000:5000`: Map port 5000 on your host to port 5000 in the container (where the Flask app runs).
*   `--env-file .env`: Load environment variables from your `.env` file.
*   `-v "$(pwd)/persistent_uploads":/app/flask_nocobase_importer/uploads`: Mount a directory from your host (`./persistent_uploads` - it will be created if it doesn't exist) to the `/app/flask_nocobase_importer/uploads` directory inside the container. This ensures that uploaded Excel files and temporary Parquet DataFrames are persisted across container restarts.
*   `--name nocobase-importer-app`: Assign a name to your running container for easier management.
*   `nocobase-importer`: The name of the image you built.

After running, the application should be accessible at `http://localhost:5000`.

### Development with Docker Compose

For a more integrated local development and testing environment, Docker Compose can be used to manage the web application, Redis, and RQ worker services.

1.  **Prerequisites**: Ensure you have Docker Compose installed.
2.  **Environment Configuration**: Create or update your `.env` file from `.env.example` with your configurations.
    *   For `DB_HOST`, if your PostgreSQL is running on the host machine from Docker's perspective:
        *   On Docker Desktop (Windows/Mac), you can often use `host.docker.internal`.
        *   On Linux, you might need to use your machine's local IP address on the Docker bridge network (e.g., `172.17.0.1` by default for the `docker0` bridge) or configure services to run on the same Docker network.
    *   If PostgreSQL is running as another Docker container (not defined in this `docker-compose.yml`), use its service name if they are on the same user-defined Docker network, or its IP address.
3.  **Start Services**: Navigate to the project root directory and run:
    ```bash
    docker-compose up --build
    ```
    This command will:
    *   Build the Docker image for the `web` and `worker` services (if not already built or if changes are detected).
    *   Start the Flask web server (using Gunicorn as per Dockerfile CMD), Redis, and an RQ worker.
4.  **Accessing the Application**: The application should be accessible at `http://localhost:5000`.
5.  **Stopping Services**: To stop all services, press `Ctrl+C` in the terminal where `docker-compose up` is running, or run from another terminal:
    ```bash
    docker-compose down
    ```

### Running the Web Container Manually (without Docker Compose)

The instructions below are for running the web application container by itself, assuming Redis and a database are accessible separately.

### Running RQ Workers Manually (for Local Development)

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

3.  **Ensure `.env` File is Present**: The worker tasks may need database credentials or other configurations defined in your `.env` file. Make sure your `.env` file (copied from `.env.example` and configured) is present in the current working directory (project root) when you start the worker.

4.  **Ensure Redis is Running**: The RQ worker needs to connect to a Redis server. Make sure your Redis server is running and accessible at the URL specified in your `REDIS_URL` environment variable (default is `redis://localhost:6379/0` if not set).

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
*   **Connection Errors to Redis**: Ensure Redis is running and accessible.

Using `python -m flask ...` can sometimes be more reliable than just `flask ...` if you have multiple Python versions or complex environments.

### Deployment with Portainer

1.  **Add Image**: You can either pull the image from a Docker registry (if you push it there) or use an image already present on the Docker host where Portainer is running (e.g., after building it locally as described above).
2.  **Create Container**:
    *   Click on "Add container".
    *   **Name**: Give your container a name (e.g., `NocoBaseImporter`).
    *   **Image**: Enter the image name (`nocobase-importer:latest` or your specific tag).
    *   **Port mapping**: Map the host port (e.g., `5000`) to the container port `5000`.
    *   **Environment variables**: Under the "Env" tab, add all the environment variables listed in `.env.example` (including `REDIS_URL`) with their appropriate values for your setup.
    *   **Volumes**: Under the "Volumes" tab, map a host path or a named Docker volume to the container path `/app/flask_nocobase_importer/uploads` to persist uploaded files. For example:
        *   Bind: Host path `/path/on/your/host/persistent_uploads` to container path `/app/flask_nocobase_importer/uploads`.
        *   Volume: Choose or create a named volume and map it to `/app/flask_nocobase_importer/uploads`.
    *   **Restart policy**: Optionally, set a restart policy (e.g., "Unless stopped" or "Always").
    *   Click "Deploy the container".

Ensure your database is configured to accept connections from the Docker container's IP address or the Docker network.

## Security Considerations

It's crucial to deploy and manage this application with security in mind. Here are some important considerations:

### HTTPS Enforcement
The application itself does not handle SSL/TLS termination. For production deployments, **always run this application behind a reverse proxy** (e.g., Nginx, Traefik, Caddy) that is configured to handle HTTPS and terminate SSL/TLS. Use tools like Let's Encrypt for free SSL certificates. Enforcing HTTPS is critical to protect data in transit.

### Secret Management
*   **`FLASK_SECRET_KEY`**: This key is used to sign session cookies and other security-sensitive tokens. The default value in `.env.example` **must be changed** to a strong, unique, and random string for your production environment. Keep this key confidential.
*   **Database Credentials**: Ensure your `.env` file containing database credentials (`DB_HOST`, `DB_USER`, `DB_PASSWORD`, etc.) is kept secure, has appropriate file permissions, and is **never committed to version control**.
*   **Environment Variables**: In general, prefer environment variables for all secrets, as demonstrated in the Docker deployment setup.

### File Uploads
*   **Size Limit**: The application is configured with a maximum file upload size (currently 16MB, set by `MAX_CONTENT_LENGTH` in `app.py`) to prevent denial-of-service attacks via excessively large uploads.
*   **Filename Sanitization**: Uploaded filenames are processed using Werkzeug's `secure_filename` to prevent directory traversal attacks or malicious filenames.
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
*   The provided `Dockerfile` and deployment instructions aim for a secure setup. Ensure your Docker host is secured and that you manage your `.env` files and any mapped volumes appropriately.
```
