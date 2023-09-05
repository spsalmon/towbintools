# towbintools!

This is the package containing all the important functions used by the Towbin Lab of the University of Bern.
Most of the code is a python translation of our old Matlab pipeline.

## Setting up a Virtual Environment

Using a virtual environment isolates your package dependencies and settings from your system Python.
Here's how you can set one up:

1. First, create a folder where the venv will be stored:

   ```bash
   mkdir ~/env_directory

   ```

2. Create the venv:

   ```bash
   python3 -m venv ~/env_directory/towbintools

   ```

3. To activate the venv:

   ```bash
   source ~/env_directory/towbintools/bin/activate

   ```

4. Whenever you are done working, you can deactivate it with:

   ```bash
   deactivate
   ```

## How to add a python venv to Jupyter

1. First, make sure Jupyter and related packages are installed:

   ```bash
   pip3 install jupyter ipykernel
   ```

2. Add your venv to Jupyter:

   ```bash
   python3 -m ipykernel install --user --name=towbintools

   ```

3. If you're using VSCode, reload VSCode and you should be able to find the kernel.

## Install the package using pip

Simply run the following command:

```bash
pip3 install towbintools
```

## Build the package and install it

1. First, make sure build is installed:

   ```bash
   pip3 install build

   ```

2. Go to the package directory, eg:

   ```bash
   cd ~/towbintools

   ```

3. Build the package:

   ```bash
   python3 -m build

   ```

4. Install the package you just built:

   ```bash
   pip3 install dist/*.whl
   ```

You're now all set!
