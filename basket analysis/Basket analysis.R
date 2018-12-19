
# Title: Basket analysis in R
# Last update: 2018.04.03
# File:  Basket analysis.R

###############
# Project Notes
###############

# Summarize project: we will use R to conduct a market basket analysis. we will discover interesting 
# relationships (or associations) between customer’s transactions and the item(s) they’ve purchased. 

###############
# Housekeeping
###############
# Clear objects if necessary
rm(list = ls())
getwd()
################
# Load packages
################
install.packages("arules")
install.packages("arulesViz")
install.packages("readr")
library(arules)
library(arulesViz)
library(readr)
###############
# Import data
##############

## Load training and test set
ElectronidexTransactions<- read.transactions("ElectronidexTransactions2017.csv", format = "basket", sep=',')

################
# Evaluate data
################
inspect (head(ElectronidexTransactions)) # You can view the transactions. Is there a way to see a certain # of transactions?
length (ElectronidexTransactions) # Number of transactions.
size (ElectronidexTransactions) # Number of items per transaction
LIST(ElectronidexTransactions) # Lists the transactions by conversion (LIST must be capitalized)
itemLabels(ElectronidexTransactions)# To see the item labels
summary(ElectronidexTransactions)

###transactions as itemMatrix in sparse format with
###9835 rows (elements/itemsets/transactions) and
###125 columns (items) and a density of 0.03506172 

###most frequent items:
### iMac                HP Laptop      CYBERPOWER Gamer Desktop            Apple Earpods        Apple MacBook Air 
###2519                     1909                     1809                     1715                     1530 

################
# Visualize data
################
itemFrequencyPlot(ElectronidexTransactions,topN =10) 
itemFrequencyPlot(ElectronidexTransactions, support = 0.1) 
image(ElectronidexTransactions[1:300]) 
image(sample(ElectronidexTransactions, 300))
             
#############
#Create rules
#############            
Rules<- apriori (ElectronidexTransactions, parameter = list(minlen=2,support = 0.005, confidence= 0.50))
inspect(Rules)

###############
#Evaluate Model
###############  
summary(Rules) 
inspect(sort(Rules, by = "lift"))

##############
#Improve Model
##############

r<-is.redundant(Rules, measure = "confidence") ##check the redundant rules. 
summary(r)
rules.pruned <- Rules[!r]
summary(rules.pruned)

inspect(sort(rules.pruned, by = "lift")[1:20])
inspect(sort(rules.pruned, by = "support")[1:20])
inspect(sort(rules.pruned, by = "confidence")[1:20])

##################
#Visualize results
################## 

plot(rules.pruned)
plot(rules.pruned[1:20], method="graph", control = list(type="items")) 
plot(rules.pruned, method="paracoord", control=list(reorder=TRUE))

########################################## 
#subsetting rules containing specific item
##########################################  

# subsetting rules containing ASUS2Monitor
ASUS2MonitorRules <- subset(rules.pruned, items %in% "ASUS 2 Monitor")
inspect(sort(ASUS2MonitorRules, by = "lift"))
summary(ASUS2MonitorRules)

# subsetting rules containing iMac

iMacRules <- subset(rules.pruned, items %in% "iMac")
inspect(sort(iMacRules, by = "lift"))
summary(iMacRules)

# subsetting rules containing Dell Desktop

DellDesktopRules <- subset(rules.pruned, items %in% "Dell Desktop")
inspect(sort(DellDesktopRules, by = "lift"))
summary(DellDesktopRules)

# subsetting rules containing HP Laptop

HPLaptopRules <- subset(rules.pruned, items %in% "HP Laptop")
inspect(sort(HPLaptopRules, by = "lift"))
summary(HPLaptopRules)

# subsetting rules iMac on RHS

iMacrhsrules <- subset(rules.pruned, rhs %in% "iMac") 
iMacrhsrules<-sort(iMacrhsrules, by = "lift")
inspect(subset(iMacrhsrules, lift > 2.5))

#     lhs                                                            rhs    support   confidence   lift   count
#[1] {ASUS 2 Monitor,Dell Desktop,Lenovo Desktop Computer}       => {iMac} 0.005185562 0.7391304  2.885807 51   
#[2] {ASUS 2 Monitor,ASUS Monitor}                               => {iMac} 0.005083884 0.7142857  2.788805 50   
#[3] {ASUS 2 Monitor,Microsoft Office Home and Student 2016}     => {iMac} 0.005185562 0.6986301  2.727681 51   
#[4] {Dell Desktop,Lenovo Desktop Computer,ViewSonic Monitor}    => {iMac} 0.006914082 0.6938776  2.709125 68   
#[5] {Apple Magic Keyboard,Dell Desktop,Lenovo Desktop Computer} => {iMac} 0.005287239 0.6842105  2.671382 52   
#[6] {Apple Magic Keyboard,ASUS Monitor}                         => {iMac} 0.006812405 0.6700000  2.615899 67   
#[7] {Acer Desktop,HP Laptop,ViewSonic Monitor}                  => {iMac} 0.006405694 0.6562500  2.562215 63   
#[8] {Acer Desktop,ASUS 2 Monitor}                               => {iMac} 0.006405694 0.6428571  2.509925 63                                                              rhs    support     confidence lift     count

summary(iMacrhsrules)

# subsetting rules HP Laptop on RHS

HPLaptoprhsrules <- subset(rules.pruned, rhs %in% "HP Laptop") 
HPLaptoprhsrules<-sort(HPLaptoprhsrules, by = "lift")
inspect(subset(HPLaptoprhsrules, lift > 3.0))

#      lhs                                                         rhs         support   confidence lift     count
#[1] {Acer Aspire,Dell Desktop,ViewSonic Monitor}             => {HP Laptop} 0.005287239 0.8125000  4.185928  52  
#[2] {Acer Aspire,iMac,ViewSonic Monitor}                     => {HP Laptop} 0.006202339 0.6630435  3.415942  61  
#[3] {Acer Desktop,iMac,ViewSonic Monitor}                    => {HP Laptop} 0.006405694 0.6363636  3.278489  63  
#[4] {Dell Desktop,Lenovo Desktop Computer,ViewSonic Monitor} => {HP Laptop} 0.006202339 0.6224490  3.206802  61  
#[5] {Computer Game,ViewSonic Monitor}                        => {HP Laptop} 0.007422471 0.6186441  3.187200  73  
#[6] {Computer Game,Dell Desktop}                             => {HP Laptop} 0.005693950 0.6086957  3.135946  56  
#[7] {Acer Aspire,ViewSonic Monitor}                          => {HP Laptop} 0.010777834 0.6022727  3.102856 106  
#[8] {Acer Desktop,Apple Magic Keyboard}                      => {HP Laptop} 0.006405694 0.5943396  3.061985  63  
#[9] {Dell Desktop,iMac,ViewSonic Monitor}                    => {HP Laptop} 0.008744281 0.5931034  3.055617  86 

summary(HPLaptoprhsrules)

# subsetting rules Lenovo Desktop Computer on RHS

LenovoDesktoprhsrules <- subset(rules.pruned, rhs %in% "Lenovo Desktop Computer") 
LenovoDesktoprhsrules<-sort(LenovoDesktoprhsrules, by = "lift")
inspect(subset(LenovoDesktoprhsrules, lift > 3.0))
#   lhs                                         rhs                        support     confidence lift     count
#[1] {ASUS 2 Monitor,Dell Desktop,iMac}       => {Lenovo Desktop Computer} 0.005185562 0.5730337  3.870732 51   
#[2] {Apple Magic Keyboard,Dell Desktop,iMac} => {Lenovo Desktop Computer} 0.005287239 0.5200000  3.512500 52   
#[3] {HP Laptop,HP Monitor,iMac}              => {Lenovo Desktop Computer} 0.005388917 0.5096154  3.442354 53   
summary(LenovoDesktoprhsrules)


# subsetting rules containing "Monitor" in the names of items
Monitorrules <- subset(rules.pruned, items %pin% "Monitor") 
Monitorrules<-sort(Monitorrules, by = "lift")
inspect(subset(Monitorrules, lift > 3.0))
#lhs                                                         rhs                       support     confidence lift     count
#[1] {Acer Aspire,Dell Desktop,ViewSonic Monitor}             => {HP Laptop}               0.005287239 0.8125000  4.185928  52  
#[2] {ASUS 2 Monitor,Dell Desktop,iMac}                       => {Lenovo Desktop Computer} 0.005185562 0.5730337  3.870732  51  
#[3] {HP Laptop,HP Monitor,iMac}                              => {Lenovo Desktop Computer} 0.005388917 0.5096154  3.442354  53  
#[4] {Acer Aspire,iMac,ViewSonic Monitor}                     => {HP Laptop}               0.006202339 0.6630435  3.415942  61  
#[5] {Acer Desktop,iMac,ViewSonic Monitor}                    => {HP Laptop}               0.006405694 0.6363636  3.278489  63  
#[6] {Dell Desktop,Lenovo Desktop Computer,ViewSonic Monitor} => {HP Laptop}               0.006202339 0.6224490  3.206802  61  
#[7] {Computer Game,ViewSonic Monitor}                        => {HP Laptop}               0.007422471 0.6186441  3.187200  73  
#[8] {Acer Aspire,ViewSonic Monitor}                          => {HP Laptop}               0.010777834 0.6022727  3.102856 106  
#[9] {Dell Desktop,iMac,ViewSonic Monitor}                    => {HP Laptop}               0.008744281 0.5931034  3.055617  86
summary(Monitorrules)


