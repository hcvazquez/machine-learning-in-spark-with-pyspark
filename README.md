# Machine Learning in Spark with Pyspark

This repository cover an special some use cases of Spark fot Machine-Learning applications using Pyspark. I will introduce Spark, just for informational purposes

## History of Spark

The Spark project started in 2009 in AMP lab in UC Berkeley. It was the time of the birth of high-computing framewors. I remember work with another high performance computing framework, GridGain, but that's another story. Initially, Spark was a research project with a focus to build a fast in-memory computing framework. Three years later in 2012 Spark had the first public release. 
As Sparks started to gain traction in the industry, it was no longer a research project, 
and to facilitate better development model and community engagement Spark has moved to Apache Foundation. 
Becoming a top level project in 2014. In the same year Spark has reached version 1.0 and two years later in 2016, 2.0. 
I find it helpful to think of Spark development in epoch terms, where every epoch has its own key ideas to explore and its own goals to achieve. 
The first epoch is the inception of Spark. 
Original development was motivated by several key observations. 
For many use cases, it was questionable if the MapReduce is an efficient model for computations. 
First, it was observed that cluster memory is usually underutilized. 
Some data sets are small enough to fit completely in cluster memory, 
while others are within a small factor of cluster memory. 
Given that memory prices are decreasing, year over year it is economically efficient to buy extra memory to feed the entire data. 
Second, there are redundant input and output operations in the MapReduce. 
For ad-hoc tasks, it is more important to reduce the completion time, 
rather than provide durability of the storage, 
because ad hoc queries generate many temporary, 
or one off data sets that could be quickly disposed. 
Third, the framework is not that composable, 
as developers would like it to be. 
For example it is tedious to reimplement joints over and over again, 
as a code reuse is complicated requiring some engineering discipline. 
Spark addresses these issues.
Many design shortcomings were fixed by introducing an appropriate composable abstraction called RDD. 
Also RDD abstraction allowed for more flexibility for the implementation and the execution layer thus, 
addressing the performance issues. 
The second development epoch, was about the integration. 
The key observation, was that typically users had several frameworks installed on their clusters, 
and each of these frameworks was used for its own purpose. 
An example here, is the MapReduce use for batch processing, 
Storm for stream processing and Elastic search for interactive exploration. 
Spark developers, try to build a unified computation framework suitable for both batch processing, 
stream processing, graphical computations and large scale machine learning. 
The effort resulted in the separation of Spark core layer consisting of basic abstractions and functions, 
and a set of Spark applications on top of the core. 
The third development epoch, which is still ongoing is driven by the wide adoption of Spark in data science community. 
Many data scientists, use specialized libraries and languages like R or Julia in their everyday work. 
These tools, use relational data models. Spark has embraced the same model in the form of Spark data frames, thus, enabling smooth and efficient integration with the data scientists tools.

Despite the many applications of Spark, as I said before, this repository cover an special use case, Machine Learning on Spark, using for that the python API implentation, Pyspark.
