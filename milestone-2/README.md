# Flask Project Setup and Run Instructions

## Prerequisites
- Python 3 installed on your system
- `pip` (Python package manager) installed

## Steps to Set Up and Run the Project

1. **Install `virtualenv`**:
   ```bash
   pip3 install virtualenv
   ```

2. **Create a Virtual Environment**:
   Navigate to the project directory then create a virtual environment:
   ```bash
   virtualenv env
   ```

3. If you can't run step 2, virtualenv is already installed but not found, locate it manually: 
    ```bash
    pip3 show virtualenv
    ```
- Once you find the path, add it to your PATH:

    ```bash
    export PATH="/path/to/virtualenv:$PATH"
    ```
    Replace /path/to/virtualenv with the directory containing the virtualenv executable.

- Then reload your shell:

    ```bash
    source ~/.zshrc
    ```

- Once virtualenv is accessible, run:
    ```bash
    virtualenv env
    ```

3. **Activate the Virtual Environment**:
   ```bash
   source env/bin/activate
   ```

4. **Install Required Packages**:
   Install the dependencies listed in `requirements.txt`:
   ```bash
   pip3 install -r requirements.txt
   ```

5. **Run the Flask Application**:
   Start the Flask development server:
   ```bash
   flask run
   ```

6. **Access the Application**:
   Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```
7. **Freeze your environment for sharing**
    After development is done:
    ```bash
    pip3 freeze > requirements.txt
    ```
    That way your friend gets the exact versions you used.

## Notes
- Ensure you are in the project directory when running the commands.
- Use the virtual environment for all Python-related commands to avoid conflicts with global packages.
- To deactivate the virtual environment, run:
  ```bash
  deactivate
  ```
