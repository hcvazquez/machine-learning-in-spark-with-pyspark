#Exploratory Data Analysis
df.describe().show()
df.groupBy('Country').count().show()
df = df.withColumnRenamed("Platform", "Search_Engine")
df.groupBy('Search_Engine').count().show()
df.groupBy('Status').count().show()
df.groupBy('Country').mean().show()
df.groupBy('Search_Engine').mean().show()
df.groupBy('Status').mean().show()
