

# Anaconda environment creation by environment.yaml

To create the right environment to launch a flask server with tensor flow and keras prediction you need to:

1. start anaconda navigator

2. launch a terminal: go to anaconda environments(left menu), select base (root), click on green arrow and select *Open Terminal*

3. write on the terminal the follow script: 

   ```
   conda env create -f {path of environment.yml}
   conda activate flask-server-env
   ```
4. to verify installation type: 
   ```
   conda list
   ```