# Imports the files library used for downloading in Colab
from google.colab import files 
import os

# 1. Zip the OHS database folder (recursively)
# This command creates the ohs_chroma_db.zip file in your Colab session.
!zip -r ohs_chroma_db.zip ohs_chroma_db/

# 2. Download the created zip file
# This command initiates the download prompt in your browser.
files.download('ohs_chroma_db.zip')