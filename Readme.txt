HOW TO RUN THE PROJECT CODE:


Step 1: Install Python 3.10+
Ensure you have Python 3.10 or higher installed on your machine. You can download it from the official Python website.

Step 2: Unzip the files
Unzip the project files into a directory on your machine.

Step 3: Install dependencies
Navigate to the project directory in your terminal and run the following command to install the required Python dependencies:

pip install -r requirements.txt

Step 4: Extract documents from the Spectrum website
To extract documents from the Spectrum website, run the following command:

python3 extract.py

By default, this will extract 50 documents. If you wish to change the number of documents to be extracted, use the --max_files option with the desired number of files:

python3 extract.py --max_files 80

Step 5: Run the inverted index script
Once the documents have been extracted, run the inverted_index.py script to process and generate the inverted index:

python3 inverted_index.py

Step 6: Run the clustering script
After generating the inverted index, run the cluster.py script to perform clustering on the extracted documents:

python3 cluster.py
This script will cluster the documents based on the provided logic and save the results.