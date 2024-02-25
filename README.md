Correlation Analysis in Scholarly Articles:
•	Used pyspark, read .json files as RDDs (size: tens of millions), preprocessed text data via regular expressions, and ntlk, applied Map-Reduce to compute normalized TF-IDF, and studied the matching degree between titles and abstracts.
•	Calculated the term frequency of each word grouped by categories, drew heat maps via seaborn and analyzed the correlation between different fields. 

To run the code, use:
spark-submit lab2.py data.jsonl stopwords.txt output.txt

Configuration:
Python 3.10.6, Spark 3.3.2, Scala 2.12.15, java version "1.8.0_341", Runs on MacOS X Version 12.5.1

Output:
output.txt, contains the miss documents, with the index accuracy, title name, and top 3 documents.

In case the environment issue, I also provide an executable .ipynb.