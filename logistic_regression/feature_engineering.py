#converting categorical data to numerical form
from pyspark.ml.feature import StringIndexer

#Indexing 
search_engine_indexer = StringIndexer(inputCol="Search_Engine", outputCol="Search_Engine_Num").fit(df)
df = search_engine_indexer.transform(df)
df.show(5,False)

from pyspark.ml.feature import OneHotEncoder

#one hot encoding
search_engine_encoder = OneHotEncoder(inputCol="Search_Engine_Num", outputCol="Search_Engine_Vector")
df = search_engine_encoder.transform(df)
df.show(3,False)
df.groupBy('Search_Engine').count().orderBy('count',ascending=False).show(5,False)
df.groupBy('Search_Engine_Num').count().orderBy('count',ascending=False).show(5,False)
df.groupBy('Search_Engine_Vector').count().orderBy('count',ascending=False).show(5,False)

country_indexer = StringIndexer(inputCol="Country", outputCol="Country_Num").fit(df)
df = country_indexer.transform(df)
df.select(['Country','Country_Num']).show(3,False)

#one hot encoding
country_encoder = OneHotEncoder(inputCol="Country_Num", outputCol="Country_Vector")
df = country_encoder.transform(df)
df.select(['Country','country_Num','Country_Vector']).show(5,False)
df.groupBy('Country').count().orderBy('count',ascending=False).show(5,False)
df.groupBy('Country_Num').count().orderBy('count',ascending=False).show(5,False)
df.groupBy('Country_Vector').count().orderBy('count',ascending=False).show(5,False)

from pyspark.ml.feature import VectorAssembler

df_assembler = VectorAssembler(inputCols=['Search_Engine_Vector','Country_Vector','Age', 'Repeat_Visitor','Web_pages_viewed'], outputCol="features")
df = df_assembler.transform(df)
df.printSchema()
df.select(['features','Status']).show(10,False)

#select data for building model
model_df=df.select(['features','Status'])
