#Exploratory Data Analysis
df.describe().select('summary','rate_marriage','age','yrs_married','children','religious').show()
df.groupBy('affairs').count().show()
df.groupBy('rate_marriage').count().show()
df.groupBy('rate_marriage','affairs').count().orderBy('rate_marriage','affairs','count',ascending=True).show()
df.groupBy('religious','affairs').count().orderBy('religious','affairs','count',ascending=True).show()
df.groupBy('children','affairs').count().orderBy('children','affairs','count',ascending=True).show()
df.groupBy('affairs').mean().show(5,True)