## RL Bootcamp Tutorial Installation Guide

Welcome to the **RL Bootcamp Tutorial**! This guide will help you set up a Python environment using `requirements.txt` to ensure you have all the necessary dependencies to run the tutorials smoothly. Follow the steps below to get started.

### Table of Contents

- [Prerequisites](#prerequisites)
- [Step 1: Install Python 3.11.9 or Higher](#step-1-install-python-3119-or-higher)
  - [On Windows](#on-windows)
  - [On macOS](#on-macos)
- [Step 2: Clone the Repository](#step-2-clone-the-repository)
- [Step 3: Create a Virtual Environment](#step-3-create-a-virtual-environment)
- [Step 4: Activate the Virtual Environment](#step-4-activate-the-virtual-environment)
  - [On Windows](#on-windows-1)
  - [On macOS/Linux](#on-macoslinux)
- [Step 5: Install Dependencies](#step-5-install-dependencies)
- [Step 6: Verify the Installation](#step-6-verify-the-installation)
- [Step 7: Deactivate the Virtual Environment](#step-7-deactivate-the-virtual-environment)
- [Additional Tips](#additional-tips)
- [Troubleshooting](#troubleshooting)
- [Further Assistance](#further-assistance)

---

### Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Git**: [Install Git](https://git-scm.com/downloads)
- **Python 3.11.9 or higher**: [Download Python](https://www.python.org/downloads/)
- **pip**: Comes bundled with Python. Verify installation with `pip --version`

---

### Step 1: Install Python 3.11.9 or Higher

#### On Windows

1. **Download the Python Installer:**

   - Visit the [Python Downloads for Windows](https://www.python.org/downloads/windows/) page.
   - Download the latest Python 3.11.x installer (e.g., `python-3.11.9-amd64.exe`).

2. **Run the Installer:**

   - Locate the downloaded `.exe` file and double-click to run it.
   - **Important:** Check the box that says `Add Python 3.11 to PATH`.
   - Click on `Install Now`.
   - Follow the on-screen instructions to complete the installation.

3. **Verify the Installation:**

   - Open Command Prompt (`Win + R`, type `cmd`, and press `Enter`).
   - Run:
     ```bash
     python --version
     ```
     You should see:
     ```
     Python 3.11.9
     ```

#### On macOS

1. **Download the Python Installer:**

   - Visit the [Python Downloads for macOS](https://www.python.org/downloads/macos/) page.
   - Download the latest Python 3.11.x installer (e.g., `python-3.11.9-macosx10.9.pkg`).

2. **Run the Installer:**

   - Locate the downloaded `.pkg` file and double-click to run it.
   - Follow the installation prompts, agreeing to the license and selecting the installation location.

3. **Verify the Installation:**

   - Open Terminal (`Command + Space`, type `Terminal`, and press `Enter`).
   - Run:
     ```bash
     python3 --version
     ```
     You should see:
     ```
     Python 3.11.9
     ```

---

### Step 2: Clone the Repository

Clone the **RL Bootcamp Tutorial** repository to your local machine using Git.

1. **Open Terminal or Command Prompt.**

2. **Run the Clone Command:**

   ```bash
   git clone https://github.com/SARL-PLUS/RL_bootcamp_tutorial.git
   ```

3. **Navigate to the Project Directory:**

   ```bash
   cd RL_bootcamp_tutorial
   ```

---

### Step 3: Create a Virtual Environment

Creating a virtual environment isolates your project's dependencies from other Python projects on your system.

1. **Run the Following Command:**

   ```bash
   python3 -m venv venv
   ```

   - **Explanation:**
     - `python3`: Specifies the Python interpreter. Use `python` if `python3` is not recognized.
     - `-m venv`: Uses the `venv` module to create a virtual environment.
     - `venv`: The name of the virtual environment directory. You can name it differently if preferred.

---

### Step 4: Activate the Virtual Environment

Before installing dependencies, activate the virtual environment.

#### On Windows

1. **Run the Activation Script:**

   ```bash
   venv\Scripts\activate
   ```

2. **Confirmation:**

   - Your command prompt should now be prefixed with `(venv)` indicating that the virtual environment is active.
     ```
     (venv) C:\Path\To\RL_bootcamp_tutorial>
     ```

#### On macOS/Linux

1. **Run the Activation Script:**

   ```bash
   source venv/bin/activate
   ```

2. **Confirmation:**

   - Your terminal prompt should now be prefixed with `(venv)` indicating that the virtual environment is active.
     ```
     (venv) your-mac:RL_bootcamp_tutorial user$
     ```

---

### Step 5: Install Dependencies

With the virtual environment activated, install the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

- **Notes:**
  - Ensure that the `requirements.txt` file is present in the project directory.
  - This command installs all packages listed in `requirements.txt` into the virtual environment.

---

### Step 6: Verify the Installation

To confirm that all dependencies are installed correctly, list the installed packages.

```bash
pip list
```

**Expected Output:**

A list of installed packages along with their versions, matching those specified in `requirements.txt`.

```
Package         Version
--------------- -------
numpy           1.23.1
pandas          1.4.2
...
```

---

### Troubleshooting

- **`pip` Not Found:**

  - Ensure that the virtual environment is activated.
  - Verify that Python and `pip` are correctly installed and added to your system's PATH.

- **Permission Issues:**

  - Avoid using `sudo` with `pip`. Instead, use a virtual environment.
  - If necessary, add the `--user` flag:
    ```bash
    pip install --user -r requirements.txt
    ```

- **Virtual Environment Activation Issues:**

  - **macOS/Linux:**
    - Ensure that the activation script has execute permissions.
    - If you encounter a "permission denied" error, run:
      ```bash
      chmod +x venv/bin/activate
      ```

  - **Windows:**
    - If you receive an execution policy error, open PowerShell as an administrator and run:
      ```powershell
      Set-ExecutionPolicy RemoteSigned
      ```
    - Then, try activating the virtual environment again.

- **Incompatible Python Version:**

  - Ensure that the active Python interpreter is **3.11.9** or higher.
  - You can specify the Python version when creating the virtual environment:
    ```bash
    python3.11 -m venv venv
    ```
    *Replace `python3.11` with the path to the desired Python executable if necessary.*

- **Missing `requirements.txt`:**

  - Ensure that you are in the correct project directory.
  - If `requirements.txt` is missing, you may need to generate it or obtain it from the repository.

---

### Further Assistance

- **Official Python Documentation:** [https://docs.python.org/3/](https://docs.python.org/3/)
- **Git Documentation:** [https://git-scm.com/doc](https://git-scm.com/doc)
- **Virtual Environments (`venv`):** [https://docs.python.org/3/library/venv.html](https://docs.python.org/3/library/venv.html)
- **Pip Documentation:** [https://pip.pypa.io/en/stable/](https://pip.pypa.io/en/stable/)
- **Community Support:**
  - [Stack Overflow](https://stackoverflow.com/)
  - [Python Discord](https://pythondiscord.com/)
  - [GitHub Discussions](https://github.com/SARL-PLUS/RL_bootcamp_tutorial/discussions)

---
