# Project demo
[https://drive.google.com/file/d/1dWnI1ht--TuJ2UAbLuOUQrsNei_lhTgx/view?usp=sharing]
# How to run

## Step 1:
- Install PostgreSQL: [https://www.postgresql.org/download/]
- YouTube Tutorial for PostgreSQL: [https://www.youtube.com/watch?v=0n41UTkOBb0&t=319s]

Open pgAdmin 4 and create a database called "eeg_db" under "Databases".
## Step 2:
- Download Model & testdata file, link:
[https://drive.google.com/drive/folders/1e3WA10IKF8SyxogZPm1diuiCQdQXAWqE?usp=drive_link]
## Step 3:
- Install Miniconda if you haven't.
**Important**: Your tensorflow version need to be 2.10
Create & activate the environment using the provided environment.yml:
```bash
conda env create -f environment.yml
conda env list
conda activate ./path
pip install tensorflow==2.10
```
If you cant create an environment using environment.yml, try:
```bash
conda create --name <my-env>
conda env list
conda activate ./path
pip install tensorflow==2.10
```
Then, install the required libraries in your activated environment.

## Step 3:

    Change your database URI in your application:
```bash
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://[username]:[password]@localhost/eeg_db'
```
    Change the model path if necessary.
```bash
    BASE_DIR =""
```
## Step 4:

Run the application:

python app.py

Done

