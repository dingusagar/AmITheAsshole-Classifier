# Data directory for all the data

## Files available:
* `pull_data.sh`
* `clean_data.py`

## Get the datasets

To get the datasets run the following from this directory (`cs7641-project/data`):
```
# change permissions and run
chmod u+x pull_data.sh
```

This will pull the data and create sym links in 'files' directory

## Run cleaner

Again while in directory (`cs7641-project/data`):
```
python3 clean_data.py files/aita_clean.csv files/train-00000-of-00001.parquet
```

Which print out and create the `cleaned_data.csv` file with the following columns:
```
'title', 'body', 'top_comment_{1,10}, 'verdict'
```