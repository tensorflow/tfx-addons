from tfx_addons.components_addons.components import RemoteZipCsvExampleGen
import os

# temp location to perform downloading and extraction
INPUT_BASE = os.getcwd()
# file url to download all the file
URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"

remote_zip_csv_example_gen = RemoteZipCsvExampleGen(
    input_base=INPUT_BASE,
    zip_file_uri=URL
)
