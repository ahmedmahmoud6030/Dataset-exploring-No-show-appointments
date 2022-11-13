#!/usr/bin/env python
# coding: utf-8

# > **Tip**: Welcome to the Investigate a Dataset project! You will find tips in quoted sections like this to help organize your approach to your investigation. Before submitting your project, it will be a good idea to go back through your report and remove these sections to make the presentation of your work as tidy as possible. First things first, you might want to double-click this Markdown cell and change the title so that it reflects your dataset and investigation.
# 
# # Project:Invistigating No-Show Appointments data set.
# 
# 
# 
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# This dataset collects information
# from 100k medical appointments in
# Brazil and is focused on the question
# of whether or not patients show up
# for their appointment. A number of
# characteristics about the patient are
# included in each row.
# 
# ● ‘ScheduledDay’ tells us on
# what day the patient set up their
# appointment.
# 
# 
# ● ‘Neighborhood’ indicates the
# location of the hospital.
# 
# ● ‘Scholarship’ indicates
# whether or not the patient is
# enrolled in Brasilian welfare
# program Bolsa Família.
# 
# ● Be careful about the encoding
# of the last column: it says ‘No’ if
# the patient showed up to their
# appointment, and ‘Yes’ if they
# did not show up.
# 
# 
# The questions I will be answering are:
# 
# 1) The likelihood showing being affected by the age of the patients?
# 
# 2) The likelihood of  showing  being affected by gender?
# 
# 3) The likelihood  of recieving SMS affecting the showing/no showing rate?
# 
# 4) Is there a specific day in which there is a high no showing rate ?
#   And
#  Is there a specific day in which there is a high showing rate?
# 
# 5)  Is there a neighbourhood with a higher average of no shows?
# 
# 6) Exploring relationship between chronic illnesses  and  (showing / not showing).
# 
# 7) Does the sms recieved differ from on neighbourhood to another? and does its relationship with showing up or not?
# 
# But before asking these questions I just answered some basic questions like the overall percentage of not showing and a summarized collective view about the data in the first visualization.
# 
# 

# In[1]:


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# >
# 
# ### General Properties

# In[2]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
df =pd.read_csv('no_show.csv')
df.head()


# In[3]:


df.shape


# In[4]:


df.tail(20)


# Here we loaded the data and saw the different columns of the dataframe there are some problems with the spelling of some words like hipertension and handcap we must change these.
# Also there are some unnescessary columns like the scheduled and appointment day and ID also are not related to our analysis that much.
# 
# Also by using the tail function we now the duration of the research  from 29-04-2016 to 07-06-2016 which is roughly 1 month and 1 week 

# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.duplicated().value_counts()


# There are no duplicated values nor null values so no need to remove them.

# In[8]:


df.nunique()


# There are two things that we need to further investigate which are the difference of patient and appointments which is mostly due to multiple appointments made by the same patients I will not be removing duplicated values of these as I believe it will affect the result in a way,so I will just leave it in the data.

# In[9]:


df['Handcap'].value_counts()


# here I guess it means after some research that 1 = handicaped in one limb, 2 = in 2 limbs and so forth.
# I don't think that 2 will be more dofferent than 3 or 4 or even 1 so I will just be grouping them into either handicapped or not.

# In[10]:


df.describe()


# Here are some summarized info about the data set

# Two things are observed here it the handcap as we said and the Age having a maximum of 115 which I don't think is accurate and min of -1 
# so I think I will remove these rows from the dataset because they are outliers.
# In a logical world there will be no baby with 0 age but I guess they mean that the baby is less than 1 year like 30 weeks or 10 weeks or others

# > **Tip**: You should _not_ perform too many operations in each cell. Create cells freely to explore your data. One option that you can take with this project is to do a lot of explorations in an initial notebook. These don't have to be organized, but make sure you use enough comments to understand the purpose of each code cell. Then, after you're done with your analysis, create a duplicate notebook where you will trim the excess and organize your steps so that you have a flowing, cohesive report.
# 
# > **Tip**: Make sure that you keep your reader informed on the steps that you are taking in your investigation. Follow every code cell, or every set of related code cells, with a markdown cell to describe to the reader what was found in the preceding cell(s). Try to make it so that the reader can then understand what they will be seeing in the following cell(s).

# ## Data cleaning and Trimming 
# I will write the description of the purpose of each code cell in  the cell below the code.

# In[11]:


df.rename(columns={'Hipertension': 'Hypertension','Handcap':'handicap','No-show':'no_show'}, inplace=True)


# First we will rename the columns that are misspelled 

# In[12]:


df.drop(['PatientId','ScheduledDay'], axis = 1, inplace = True)


# Next we remove the columns we won't be needing in our analysis.

# In[13]:


df.head()


# In[14]:


df.handicap[df['handicap'] >= 1] = '1'
df.handicap[df['handicap'] == 0 ] = '0'


# In[15]:


df['handicap'].value_counts()


# We will add all the hadicapped to (0,1) in which 1,2,3 will be = 1 and 0=0  this will be easier if we used it in our analysis.

# In[16]:


df.head()


# In[17]:


df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df['day'] = df['AppointmentDay'].dt.day_name()
    
df.head()


# Because I want to know if there is a specific day that people would go to more than others I will just create a new column for days from the Appointment Day column.
# 

# In[18]:


df.drop('AppointmentDay',axis = 1, inplace = True)


#  We will drop the AppointmentDay as it is irrelevant our data analysis.
#  

# In[19]:


df.loc[df['Age'].isin([-1,115])].index


# In[20]:


df.drop([63912, 63915, 68127, 76284, 97666, 99832], axis=0, inplace=True)


# In[21]:


df.head(40)


# We removed the 'Age' outliers in our dataset 

# In[22]:


df.replace('No',0, inplace = True)
df.replace('Yes',1, inplace = True)
df.head(20)


# Converting the Yes and NO responses into 0 and 1 to be simpler and easier to plot in which the 1 is yes or didn't show , and the 0 is the No or did show.  

# In[23]:



noshow = df.no_show == True
show = df.no_show == False
df[show].head()


# I will seperate the '0' and the '1' in the no_show column by using a mask to just filter the data easier.

# In[24]:


sms = df.SMS_received == True
nosms =df.SMS_received == False


# Separated people who recieved sms from people that didn't to facilitate my exploration.
# 

# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# 
# ### We look at the visuals of all the data before asking any question.

# In[25]:


# Use this, and more code cells, to explore your data. Don't forget to add
#   Markdown cells to document your observations and findings.
df.hist(figsize= [15,8]);


# Numerical summary about the data collected showing the no. of people how have chronic diseases(Diabetes, Hypertension), Age, and other displayed graph of course this summary doesn't show things like: the show or no show , handicaped , and neighborhood. so it is not conclusive data just seeing in it the numbers according to each column. 

# ## What was the percentage of people not showing from the total appointments made?

# In[26]:


df.describe()


# In[27]:


def genderplot(gender, title_name):
    gender.plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8),legend=True,title=title_name,labeldistance=None);


# In[28]:



y=df['no_show'].value_counts()
genderplot(y,'No show vs Show Percentage')


# This diagram shows us that only 20% or 1/5 of patients didn't show to their appointments.

# # Exploring factors that will affect the show and no-show rate.

# ## The likelihood showing being affected by the age of the patients?

# In[29]:


df.groupby('Age').no_show.mean().plot(kind = 'bar', figsize=(20,20));


# We see at this graph that infants and children have a low chance of not showing up compared to the young patients (10-31 ) years then we can see a decline in the no show rate in the middle aged patients  and it reaches its lowest no show rate in patients who are in there 60s then the rate increases again , but still lower than that of young patients

# In[30]:


def plotting(x):
    x[show].hist(alpha=0.5, bins=20, label='show', color = 'lightgreen')
    x[noshow].hist(alpha=0.5, bins=20, label='No show', color = 'darkred')   
    plt.legend();


# In[31]:


age = df.Age
plotting(age)


# Mostly as we said that only 20% of patients didn't show up and that the rating of showing up is highest at birth of the early years of life and it also increase in the middle ages around (45-60) and the rates at the end are both low probably because to few people survive to the late 80s and 90s.
# While the highest group age with the no show percentage is probably the  around teen years and around (25-40) we can see that the ratio of showing to not showing there is high compared to infants and children.

# ###  The likelihood of  showing  being affected by gender?
# Meaning is there a gender with a higher number of showing compared to their no show.

# In[32]:


total =  df['Gender'].value_counts()
genderplot(total,'Percentage of Males and Females')


# In[33]:


q= df.groupby('no_show')["Gender"].value_counts()
genderplot(q,'No show vs Show in females')


# In[34]:


no_gender= df[noshow].Gender.value_counts()
genderplot(no_gender, 'No show Females Vs Males')


# In[35]:


show_gender=df[show].Gender.value_counts()
genderplot(show_gender,' Showed Females Vs Males')


# As we saw in the first pie chart that the proportions of M to F in showing is nearly equal to that in not showing.
# The last 2 pie charts just confirm my analysis.
# Therefore: although the females were more than males in the showing and not showing probably due to the fact that number of females to total number of patients is 65 which is a rate that is consistant in all of the analysis. The gender doesn't correlate to the rate of not showing or showing

# ## The likelihood  of recieving SMS affecting the showing/no showing rate?
# In this question I would like to answer 2 more subquestions.
# #### How many people  out of all the people who showed received sms?
# #### How many people out of all who didn't show up received sms?

# In[36]:


df[sms].no_show.value_counts()


# In[37]:


def smsplot(smsx, title, xlable, ylable, type):
    smsx.plot(kind= type, label =title );  
    plt.title(title, fontsize=14)
    plt.xlabel(xlable , fontsize=12)
    plt.ylabel(ylable, fontsize=12)
    plt.show()


# In[38]:


smsx = df[sms].no_show.value_counts()
smsplot(smsx, 'Show vs no show in people who received sms','Showed(0) and No show(1)','No. of Patients','bar')


# In this graph it is shown that people showed received more sms message than the no show people, but I believe that this is expected especially if you look at the perentage it tells you that the rate of messages is probably high in the no show patients as the percentage of this is close to 28% compared to the 20% percentage of the actual show vs no show.

# In[39]:


df[sms].no_show.value_counts()


# In[40]:


x= (9784/df[sms].no_show.count())*100
print(x)


# In[41]:


shows = df[show].SMS_received.value_counts()
smsplot(shows,"Number of people who Showed in relation to recieving sms","Didn't receive(0) and received(1)",'No. of Patients','bar')


# In[42]:


noshows = df[noshow].SMS_received.value_counts()
smsplot(noshows,"Number of people who didn't show in relation to recieving sms","Didn't receive(0) and received(1)",'No. of Patients','bar')


# In[43]:


df[noshow].SMS_received.value_counts()[1]/df[noshow].SMS_received.count()


# In[44]:


df[show].SMS_received.value_counts()[1]/df[show].SMS_received.count()


# As shown here there is no relationship between the sms recieved and the patient's showing rate in fact it rate of recieved sms it higher in people not showing than people showing as it is shown above.

# In[45]:


df.head()


# ## Relation between having the Scholarship and Showing up for the appointment.

# In[46]:


df[show].Scholarship.value_counts()[1],df[noshow].Scholarship.value_counts()[1]


# In[47]:


percentage_show = (df[show].Scholarship.value_counts()[1]/df[show].Scholarship.count())*100
percentage_noshow =(df[noshow].Scholarship.value_counts()[1]/df[noshow].Scholarship.count())*100
print(round(percentage_show),"% of the people who showed had the scholarship, while ",round(percentage_noshow),"% of people who didn't show had a scholarship")


# The result of the analysis states that there is no correlation between having a scholarship in increase in the rate of showing up. 

# I couldn't display the results because I don't know how every time I try to it justs loads and nothing happen.

# ## Is there a specific day in which there is a high no showing rate ?
# ## And
# ## Is there a specific day in which there is a high showing rate?

# In[48]:


day_show=df[show].day.value_counts()
day_noshow = df[noshow].day.value_counts()


# In[49]:


genderplot(day_show,'  Total number attendence of each day')


# In[50]:


genderplot(day_noshow,'The total number of no shows in each day.')


# In[51]:


df[show].day.value_counts()["Saturday"]


# In[52]:


df[noshow].day.value_counts()["Saturday"]


# One problem with this set is that there was so little appointments made  on saturday  further research is required to conclude the result of this anomaly.
# The data result tell us that Sunday is the day with the highest number of show and no show and probably the day with the highest appointments. 
# Therefore there is no day with clearly higher rate of shows than no shows.

# ## Is there a neighbourhood with a higher average of no shows?

# In[53]:


df[show].Neighbourhood.value_counts().plot(kind='bar',alpha=0.5, color= 'blue', label='show',figsize=(35,35))
df[noshow].Neighbourhood.value_counts().plot(kind='bar',alpha=0.5, color= 'green', label='no show',figsize=(30,30))
plt.xlabel("Neighbourhoods", fontsize = 25)
plt.ylabel('No. of Patients', fontsize = 25 )
plt.legend(fontsize = 25)
plt.xticks(size = 20)
plt.yticks(size = 20);


# In[54]:


## Taking a list of the neighbourhoods(NB) and returning 2 lists 1 with NB that shows a high correlation with not showing 
## and the other for the high correlation with showing. This list is missing to values which are ['ILHAS OCEÂNICAS DE TRINDADE', 'PARQUE INDUSTRIAL']


df['Neighbourhood'].unique()
n = ['JARDIM DA PENHA', 'MATA DA PRAIA', 'PONTAL DE CAMBURI',
       'REPÚBLICA', 'GOIABEIRAS', 'ANDORINHAS', 'CONQUISTA',
       'NOVA PALESTINA', 'DA PENHA', 'TABUAZEIRO', 'BENTO FERREIRA',
       'SÃO PEDRO', 'SANTA MARTHA', 'SÃO CRISTÓVÃO', 'MARUÍPE',
       'GRANDE VITÓRIA', 'SÃO BENEDITO', 'ILHA DAS CAIEIRAS',
       'SANTO ANDRÉ', 'SOLON BORGES', 'BONFIM', 'JARDIM CAMBURI',
       'MARIA ORTIZ', 'JABOUR', 'ANTÔNIO HONÓRIO', 'RESISTÊNCIA',
       'ILHA DE SANTA MARIA', 'JUCUTUQUARA', 'MONTE BELO',
       'MÁRIO CYPRESTE', 'SANTO ANTÔNIO', 'BELA VISTA', 'PRAIA DO SUÁ',
       'SANTA HELENA', 'ITARARÉ', 'INHANGUETÁ', 'UNIVERSITÁRIO',
       'SÃO JOSÉ', 'REDENÇÃO', 'SANTA CLARA', 'CENTRO', 'PARQUE MOSCOSO',
       'DO MOSCOSO', 'SANTOS DUMONT', 'CARATOÍRA', 'ARIOVALDO FAVALESSA',
       'ILHA DO FRADE', 'GURIGICA', 'JOANA D´ARC', 'CONSOLAÇÃO',
       'PRAIA DO CANTO', 'BOA VISTA', 'MORADA DE CAMBURI', 'SANTA LUÍZA',
       'SANTA LÚCIA', 'BARRO VERMELHO', 'ESTRELINHA', 'FORTE SÃO JOÃO',
       'FONTE GRANDE', 'ENSEADA DO SUÁ', 'SANTOS REIS', 'PIEDADE',
       'JESUS DE NAZARETH', 'SANTA TEREZA', 'CRUZAMENTO',
       'ILHA DO PRÍNCIPE', 'ROMÃO', 'COMDUSA', 'SANTA CECÍLIA',
       'VILA RUBIM', 'DE LOURDES', 'DO QUADRO', 'DO CABRAL', 'HORTO',
       'SEGURANÇA DO LAR', 'ILHA DO BOI', 'FRADINHOS', 'NAZARETH',
       'AEROPORTO']
l1=[]
l2 =[]
for i in n:
    s=(df[noshow].Neighbourhood.value_counts()[i]/df[show].Neighbourhood.value_counts()[i])* 100
    print(round(s), "% of the appointments made in", i , "didn't show up")
    if round(s) >= 30:
        l1.append(i)
    elif round(s)< 20:
        l2.append(i)
        
    
    
  


# In[55]:



df.query('Neighbourhood == "PARQUE INDUSTRIAL"')


# In[56]:


df.query('Neighbourhood == "ILHAS OCEÂNICAS DE TRINDADE"')


# In[57]:


print(l1, ":these neighbourhoods are the ones with the highest rate of no shows")
print(l2, "\n :these neighbourhoods are the ones with highest rate of shows")


#  Taking a list of the neighbourhoods(NB) and returning 2 lists 1 with NB that shows a high correlation with not showing 
#  and the other for the high correlation with showing. This list is missing to values which are ['ILHAS OCEÂNICAS DE TRINDADE', 'PARQUE INDUSTRIAL'] as the first on all of them showed and the latter was only one appointment and did not show up therefore cannot be used to analyze the data.
# 

# ##  Exploring relationship between chronic illnesses  and  (showing / not showing).

# In[58]:


df[noshow].groupby(["Diabetes","Hypertension"]).count().plot(kind='bar',alpha=0.5, color= 'blue', legend= False,figsize=(10,10))
df[show].groupby(["Diabetes","Hypertension"]).count().plot(kind='bar',alpha=0.5, color= 'green',legend= False,figsize=(10,10))

plt.xticks(size = 10)
plt.yticks(size = 10);


# It seems that there is no correlation between people having chronic illnesses and them not shwoing more or even showing more. As by graph and some calcutions they seem to have similar percentage in each column. They even have the same decrease in the (1,1) column.

# In[59]:


df[noshow].groupby(["Diabetes","Hypertension"]).count()


# In[60]:


df[show].groupby(["Diabetes","Hypertension"]).count()


# ## Does the sms recieved differ from on neighbourhood to another? and does its relationship with showing up or not?

# In[63]:


df[show].groupby(["Neighbourhood"])["SMS_received"].mean().plot(kind='bar',alpha=0.5, color= 'blue', label='show',figsize=(35,35))
df[noshow].groupby(["Neighbourhood"])["SMS_received"].mean().plot(kind='bar',alpha=0.5, color= 'green', label='no show',figsize=(30,30))
plt.xlabel("Neighbourhoods", fontsize = 25)
plt.ylabel('mesn of the no of sms received', fontsize = 25 )
plt.legend(fontsize = 25)
plt.xticks(size = 20)
plt.yticks(size = 20);


# In the majority of the neighbourhood less than 50% of the people didn't receive SMS and that the rate of receiving sms is higher in people not showing except in 2 neighbourhoods , but these 2 didn't have any people who received SMS and didn't show up.
# Further investigation is required to identify the cause of this.

# most of the data fall below the 0.5 which means that on average the patients in these neighborhood didn't receive sms and didn't show which in my opinion is still weird because they are recieving SMS and still not showing in most of the neighbourhoods except in few neighbourhoods where people that received sms did go. In my opinion I find that receiving sms did not affect the showing rate as it should but the rate of receiving is higher in people not showing.Therefore there is weirdly enough a negative correlation between  

# <a id='conclusions'></a>
# ## Conclusions
# On analyzing the data set of the No-show appointments we find out that upon asking and asnwering some questions that there are factors that likely affect the increase of people showing and not showing.
# We found out that there are some neighbourhoods ['ANDORINHAS', 'PRAIA DO SUÁ', 'ITARARÉ', 'SANTA CLARA', 'SANTOS DUMONT', 'CARATOÍRA', 'JESUS DE NAZARETH', 'ILHA DO PRÍNCIPE', 'SANTA CECÍLIA', 'HORTO'] that have a higher rate than normal in not showing, while others['JARDIM DA PENHA', 'SANTA MARTHA', 'SOLON BORGES', 'MÁRIO CYPRESTE', 'DE LOURDES', 'DO CABRAL', 'ILHA DO BOI', 'AEROPORTO']  have a lower rate in not showing. 
# Also Certain age groups like infants and middle aged patients are likely to show up while the patients in their teens and up to the 40s are likely not to show up for the appointment.
# 
# In the last question asked about how SMS_received affect the showimg and not showing in every neighbourhood I find it strange that there is a correlation in in which in most of the neighbourhoods that people who received SMS where more likely to not show for their appointments.
# 
# 
# While finding out that factors such as Gender, Chronic diseases, SMS_received has no correlation on how likely a patient will show or not show.
# Some limitation I have contacted was the data collected on saturday also that I believe that some data is missing from some neighbourhoods['ILHAS OCEÂNICAS DE TRINDADE', 'PARQUE INDUSTRIAL'] which means that either they closed at the beginning of the data collection process or they stoped reporting their data. These points will require further data collection and inquery. 
# Finally I would like to say that solely based on correlation we should not conclude causation.
# 

# ## References:
# 
# Stack overflaw "https://stackoverflow.com/"
# for helping me find some solutions to problems encountered.
# 
# Udacity classroom.
# 
# Eg FWD online communinty "https://nfpdiscussions.udacity.com/"
# 
# W3schools "https://www.w3schools.com/python/matplotlib_histograms.asp"
# 
# "https://www.statology.org/matplotlib-histogram-color/"
# 
# "https://www.geeksforgeeks.org/"
# 
# I was using these websites just get know mainly how to write a function or search on how to plot or how to excute something and then find out about a new function.

# In[ ]:





# In[ ]:




