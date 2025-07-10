# towbintools!

This is the package containing all the important functions used by the Towbin Lab of the University of Bern.
Most of the code is a python translation of our old Matlab pipeline.

This package goes hand in hand with our modular pipelining tool : <https://github.com/spsalmon/towbintools_pipeline>

Documentation : <https://towbintools.readthedocs.io/en/latest/towbintools.html>
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
