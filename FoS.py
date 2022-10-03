import numpy                                      as np
from sklearn.linear_model         import LinearRegression
from scipy.stats                  import t        as tstat
import matplotlib.pyplot                          as plt
%matplotlib inline
from scipy.stats                  import f_oneway as ANOVA
from statsmodels.stats.libqsturng import qsturng
from   matplotlib.pyplot          import figure

#------------------------------------------------------------------------
#Nice plot function
#------------------------------------------------------------------------
def nplot(xlabel, ylabel, fontsize = 20, labelsize = 16, bigfig = 'off'):
    #LaTeX options
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')
    #plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    if bigfig == 'off':
        fig = plt.figure(figsize=(9,7))
    else:
        fig = plt.figure()
    
    ax  = fig.add_subplot(1,1,1)
    
    ax.set_xlabel(xlabel, fontsize = fontsize)
    ax.set_ylabel(ylabel, fontsize = fontsize)
    ax.tick_params(labelsize = labelsize)
    fig.tight_layout()
    
    plt.grid()
    plt.minorticks_on()
    plt.tick_params(axis  = 'y', which = 'minor', bottom = False)
    plt.tick_params(which = 'major', length = 7)
    plt.tick_params(which = 'minor', length = 4)



#Raw Data from https://rankings.thefire.org/rank
#--------------------------------------------------------------------
Ratio = np.array([4,1/1.3,1/1.1, 1/1.9, 1/1.2, 5.2, 3.7, 1.8, 1.6, 2.4, 1.6, 5.6, 3.4, 1/1.2, 1.6, 1.6, 3.1, 2.3, 1.9, 4.2, 2.2, 1/1.1, 1/1.5, 2.7, 1/1.1, 3.4, 4, 2.5, 3, 4.7, 1.6, 2.8,1.8,2.4,2.5,\
                  4.3,6,1.7,1/1.1, 1/1.6, 1/1.1, 6.7, 12.1, 3.7, 1/1.9, 1/2.1, 1.3, 2,1/1.9,1.1,2.4,4.9,1,10.9,1/1.7,2.5,5.6,1.1,1.8,2.3,4.6,7,3.8,1.2,1.9,1.9,1.5,1.6,11.6,1.3,2,2.5,2.1,3.3,1.8,3.2,\
                  3.5,3.4,1.1,2,1.1,5.5,4.3,4.6,2.4,2.4,2.3,27,1.3,6.2,1,1.1,9.2,2.2,2.7,1.8,2.1,2.8,4.8,11.1,1.2,1.6,2.5,3,6.3,4,1.2,2.7,3.9,3.2,3.6,2.8,4,7.8,66,2,6.4,2.4,4.5,5.2,4.6,3.2,2.6,7.3,\
                  6.4,1.1,24.5,7.8,1.4,5.3,1.6,3.2,2.1,8.4,14.4,4.5,7.6,3.2,1.2,2.9,3,1.5,1.6,1.6,2.9,2.7,6.9,5.2,7.5,2,8.2,5.5,2.1,4.2,3.3,7.8,11.7,3.2,65,8.9,6.1,3.1,4,8.1,3.9,2.8,12.5,28,3.1,5.3,\
                  4.8,12.4,2.4,7.9,6.6,3.7,3.4,3.3,2.1,3.4,5.2,19.4,3.9,1.4,13.9,5.3,60,5.6,6.8,10.6,7.3,13.7,4.8,2.6,41.5,28.8,6.4,5.2,11.8,3.3,2.3,5.9,6.8,1/10.3,1.6,1/1.6,1/1.2,1.8])

Score = np.array([77.92,76.20,75.81,74.72,74.35,72.65,68.72,68.50,67.93,67.42,66.5,66.24,65.78,65.73,65.54,65.19,64.79,64.47,64.13,62.75,62.47,62.46,62.45,62.38,61.91,61.72,60.30,59.49,59.38,58.12,57.88,\
                 57.30,56.91,56.63,56.40,56.33,56.16,55.96,55.14,54.59,54.59,54.22,54.22,53.99,53.91,53.88,53.82,53.68,53.56,53.39,53.12,52.99,52.98,52.65,52.56,52.46,52.37,52.05,51.81,51.80,51.71,51.64,\
                 51.52,51.38,51.28,51.27,51.26,51.01,50.91,90.91,50.39,50.36,50.34,50.32,50.20,50.09,49.87,49.75,49.65,49.59,49.50,49.03,48.99,48.66,48.6,48.55,48.35,48.33,48.18,48.07,47.76,47.57,47.55,\
                 47.39,47.04,47.01,46.90,46.88,46.86,46.67,46.57,46.52,46.48,46.45,46.16,45.94,45.56,45.29,45.15,45.03,45.02,45.02,44.98,44.87,44.57,44.31,44.15,44.09,43.57,43.44,43.43,43.32,42.68,42.57,\
                 42.52,42.48,42.38,41.96,41.71,41.58,41.48,40.81,40.63,40.53,40.48,40.45,40.43,40.32,40.31,39.85,39.80,39.45,39.42,39.35,39.27,39.17,39,38.98,38.52,38.43,38.36,37.82,37.78,37.63,37.56,\
                 37.33,37.29,36.85,36.7,36.53,36.37,36.25,36.18,36.16,36.07,35.83,35.77,35.77,35.32,34.52,34.38,34.32,34.29,34.18,33.85,33.65,33.64,33.53,33.48,32.75,32.38,31.93,31.33,31.29,29.55,29.27,\
                 28.86,28.61,27.33,27.31,27.04,26.92,26.90,26.5,26.35,23.51,23.09,22.65,21.51,20.48,18.60,14.32,9.91,57.51,37.59,36.43,36.29,32.19])



#Find the log of the ratio
#--------------------------------------------------------------------
logR = np.log(Ratio)



#Find the liberal, neutral, and conservative universities
#---------------------------------------------------------------------
C = np.where(logR < 0)[0]   #Where the conservative universities are
L = np.where(logR > 0)[0]   #Where the liberal universities are
N = np.where(logR == 0)[0]  #Where the neutral universities are



#Perform a linear regression on the log-Ratio data with Sklearn
#----------------------------------------------------------------------
model = LinearRegression()
model.fit(Score.reshape((-1, 1)), logR)
model = LinearRegression().fit(Score.reshape((-1, 1)), logR)
B1    = model.coef_[0]         #Slope Parameter
B0    = model.intercept_       #Intercept Parameter

yhat  = lambda Score: (B1*Score + B0)  #Make the regression line as a function



#Calculate and define important statistical parameters
#-----------------------------------------------------------------------
n   = len(Score)                                  #Number of datapoints
SSE = sum((logR-yhat(Score))**2)                  #Sum of Square Error
MSE = SSE/(n-2)                                   #Mean Square Error
SE  = np.sqrt(MSE)                                #Standard Error of the estimate
Sxx = sum((Score-np.mean(Score))**2)              #Sum of the square of the difference with respect to Score
Syy = sum((logR-np.mean(logR))**2)                #Sum of the square of the difference with respect to logR
Sxy = sum(Score*logR) - sum(Score)*sum(logR)/n    #Sum of the square of the difference with respect to Score and logR


#Perform B1 hypothesis testing
#------------------------------------------------------------------------
sB1 = SE/np.sqrt(Sxx)                    #Standard error associated with the slope (B1)
t   = B1/sB1                             #Test statistic
p   = tstat.sf(abs(t), df=n-2)           #P-value associated with H0: B1 = B10

alpha = 0.01
if p < alpha:
    Rej = ["True"]
else:
    Rej = ["False"]

print(f"""
H0: B1 = B10 hypothesis test (alpha = {alpha})
----------------------------------------------------
test statistic        p-value        Reject null? 
{t:0.2f}                 {p:0.1e}        {Rej[0]}
""")



#Calculate Sample Correlation Coefficient
#------------------------------------------------------------------------
r = Sxy/(np.sqrt(Sxx)*np.sqrt(Syy))

if abs(r) <= 0.5:
    Correl = ["Weak Correlation"]
elif abs(r) >= 0.8:
    Correl = ["Strong Correlation"]
else:
    Correl = ["Moderate Correlation"]
    

print(f"""
Sample Correlation Coefficient:
---------------------------------------------
r   = {r:0.2f}   ({Correl[0]})

r^2 = {r**2:0.2f}
""")



#Perform rho hypothesis testing on the sample correlation coefficient
#------------------------------------------------------------------------

v = 0.5*np.log((1 + r)/(1 - r))  #Bivariate normal distribution of rho

def CIrho(v):
    z = 1.96
    c = v + z/np.sqrt(n-3), v- z/np.sqrt(n-3)
    return (np.exp(2*c[0]) - 1)/(np.exp(2*c[0]) + 1), (np.exp(2*c[1]) - 1)/(np.exp(2*c[1]) + 1)   #95% CI for rho

print(f"""
95% CI for Sample Correlation Coefficient:
---------------------------------------------
ρ = ({CIrho(v)[1]:0.2f}, {CIrho(v)[0]:0.2f})
(Moderate to Weak Correlation)

""")




#Calculate Confidence and Prediction Intervals
#------------------------------------------------------------------------

def CI(Score, alpha = 0.05):
    SORT = np.array([sorted(Score)])[0]
    Y = yhat(SORT)
    ta = tstat.ppf(1-alpha/2, n-2)
    return Y+ta*SE*np.sqrt(1/n + (SORT - np.mean(SORT))**2/Sxx), Y-ta*SE*np.sqrt(1/n + (SORT - np.mean(SORT))**2/Sxx)

def PI(Score, alpha = 0.05):
    SORT = np.array([sorted(Score)])[0]
    Y = yhat(SORT)
    ta = tstat.ppf(1-alpha/2, n-2)
    return Y+ta*SE*np.sqrt(1+1/n + (SORT - np.mean(SORT))**2/Sxx), Y-ta*SE*np.sqrt(1+1/n + (SORT - np.mean(SORT))**2/Sxx)


#-----------------------------------------------------------------------------------------------
#Split data into heavy, moderate, and mild categories
#-----------------------------------------------------------------------------------------------

#Set arbitrary limits on mild, moderate, and heavy
Mild_limit  = 1.5  #1.5:1 student ratio
Mod_limit   = 3    #3:1 student ratio
Heavy_limit = 7    #7:1 student ratio

#Find where these are in the data
WL = np.where((-logR <= -np.log(Mild_limit)) & (-logR > -np.log(Mod_limit)))[0]   #Weak liberal schools
ML = np.where((-logR <= -np.log(Mod_limit)) & (-logR > -np.log(Heavy_limit)))[0]  #Moderate liberal schools
SL = np.where((-logR <= -np.log(Heavy_limit)))[0]                                 #Strong liberal schools
Cen = np.where((-logR > -np.log(Mild_limit)) & (-logR < np.log(Mild_limit)))[0]   #Neutral/central schools
WC = np.where((-logR >= np.log(Mild_limit)) & (-logR < Mod_limit))[0]             #Weak conservative schools
SC = np.where((-logR > np.log(Heavy_limit)))[0]                                   #Strong conservative schools

#Concatenate data into an array
data = [Score[SL], Score[ML], Score[WL], Score[Cen], Score[WC]]
names = [f"> {Heavy_limit}:1", f"({Mod_limit} to {Heavy_limit}):1", f"({Mild_limit} to {Mod_limit}):1","Balanced", f"1: >{Mild_limit}"]



#Perform Single-Factor ANOVA test on split data
#------------------------------------------------------------------------

ts, pv = ANOVA(Score[SL], Score[ML], Score[WL], Score[WC])

alpha2 = 0.01
if pv < alpha2:
    Rej2 = ["True"]
else:
    Rej2 = ["False"]

print(f"""
Single Factor ANOVA (alpha = {alpha2})
----------------------------------------------------
test statistic        p-value        Reject null?
{ts:0.2f}                  {pv:0.2e}       {Rej2[0]}
""")



#Perform Tukey Analysis
#------------------------------------------------------------------------

#Sum of Square Errors with respect to the mean
SSE_Tukey = 0
I = np.shape(data)[0]
for i in range(I):
    Xbari = np.mean(data[i])
    for j in range(len(data[i])):
        Xij = data[i][j]
        SSE_Tukey += (Xij - Xbari)**2
        
#Mean square error with respect to the mean
MSE_Tukey = SSE_Tukey/(n-I)

#Desired alpha value
alpha3 = 0.15

#Critical Value for Studentized Range Distribution
Q = qsturng(1- alpha3, I, n-I)

#Find wij factor
ws = np.zeros(I-1)
for i in range(len(ws)):
    Ji = len(data[i])
    Jj = len(data[i+1])
    ws[i] = Q*np.sqrt(MSE_Tukey/2*(1/Ji + 1/Jj))

#Find the difference in xbar for each grouping
xbar = np.zeros(I)
for i in range(len(xbar)):
    xbar[i] = np.mean(data[i])

dxbar = xbar[1:] - xbar[:-1]
Diff = dxbar - ws

#Develop rejection criteria
Reject = []

for i in range(len(Diff)):
    if Diff[i] > 0:
        Reject = np.append(Reject, "True")
    else:
        Reject = np.append(Reject, "False")
        
print(f"""
GROUP 1       GROUP 2     Δx12             wij-value      Reject? (alpha = {alpha3})
-------------------------------------------------------------------------------
SL            ML          {dxbar[0]:0.2f}             {ws[0]:0.2f}           {Reject[0]}
ML            WL          {dxbar[1]:0.2f}             {ws[1]:0.2f}           {Reject[1]}
WL            N           {dxbar[2]:0.2f}             {ws[2]:0.2f}           {Reject[2]}
N             WC          {dxbar[3]:0.2f}             {ws[3]:0.2f}          {Reject[3]}
""")


#---------------------------------------------------------------------------------------------
#PLOTTING
#---------------------------------------------------------------------------------------------

#Linear Scatter Plot
nplot("Overall Free Speech Score", "Liberal:Conservative Ratio")
plt.title("Linear Scatter Plot", fontsize = 22)
plt.scatter(Score[C], Ratio[C], color = 'r', marker = '^', label = 'Conservative', s = 20)
plt.scatter(Score[L], Ratio[L], color = 'b', label = 'Liberal', s = 20)
plt.scatter(Score[N], Ratio[N], color = 'k', marker = 's', label = 'Balanced', s = 20)
plt.axhline(y = 1, color = 'k')
plt.legend(fontsize = 15);


#Semilogx Scatter Plot with CI and PI
alpha = 0.05
nplot("Overall Free Speech Score", "Log Liberal:Conservative Ratio")
plt.title("Semilogx Scatter Plot with CI and PI", fontsize = 22)
plt.scatter(Score[C], logR[C], color = 'r', marker = '^', label = 'Conservative', s = 20)
plt.scatter(Score[L], logR[L], color = 'b', label = 'Liberal', s = 20)
plt.scatter(Score[N], logR[N], color = 'k', label = 'Balanced', s = 20)
plt.plot(Score, yhat(Score), color = 'g')
plt.axhline(y = 0, color = 'k')
plt.plot(sorted(Score), CI(Score, alpha = alpha)[0], color = 'k', linestyle = '-.', label = f'{(1-alpha)*100:0.0f}\% CI')
plt.plot(sorted(Score), CI(Score, alpha = alpha)[1], color = 'k', linestyle = '-.')
plt.plot(sorted(Score), PI(Score, alpha = alpha)[0], color = 'k', linestyle = '--', label = f'{(1-alpha)*100:0.0f}\% PI')
plt.plot(sorted(Score), PI(Score, alpha = alpha)[1], color = 'k', linestyle = '--')
plt.legend(fontsize = 15);

#Negative Flipped x-y semilog axis
alpha = 0.05
nplot("Log Conservative:Liberal Ratio", "Overall Free Speech Score")
plt.title("Negative Flipped x-y Semilog Scatter Plot", fontsize = 22)
plt.scatter(-logR[C], Score[C], color = 'r', marker = '^', label = 'Conservative', s = 20)
plt.scatter(-logR[L], Score[L], color = 'b', label = 'Liberal', s = 20)
plt.scatter(-logR[N], Score[N], color = 'k', marker = 's', label = 'Balanced', s = 20)
plt.plot(-yhat(Score), Score, color = 'g')
plt.axvline(x = 0, color = 'k')
plt.plot(-CI(Score, alpha = alpha)[0], sorted(Score), color = 'k', linestyle = '-.', label = f'{(1-alpha)*100:0.0f}\% CI')
plt.plot(-CI(Score, alpha = alpha)[1], sorted(Score), color = 'k', linestyle = '-.')
plt.plot(-PI(Score, alpha = alpha)[0], sorted(Score), color = 'k', linestyle = '--', label = f'{(1-alpha)*100:0.0f}\% PI')
plt.plot(-PI(Score, alpha = alpha)[1], sorted(Score), color = 'k', linestyle = '--')
plt.legend(fontsize = 15);



#Boxplot of categorized data
names = [r"$> 7:1$", r"(3 to 7):1", "(1.5 to 3):1","Balanced", "1: $>1.5$"]
fig = plt.figure(figsize = (9,7))
ax = fig.add_subplot(111)
plt.title("Boxplot", fontsize = 22)
ax.boxplot(data)
ax.set_xticklabels(names, fontsize = 15, rotation = 20)
plt.ylabel("Overall Free Speech Score", fontsize = 15)
ax.tick_params(labelsize = 15)
plt.xlabel("Liberal:Conservative Ratio", fontsize = 15)