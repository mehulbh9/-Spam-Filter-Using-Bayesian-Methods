import numpy as np
import matplotlib.pyplot as plt
import util

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    ### TODO: Write your code here
    
    #initialization
    total_male_number, total_female_number = 0, 0
    sum_height_male, sum_weight_male, sum_height_female, sum_weight_female = 0, 0, 0, 0
    list_height_male, list_weight_male, list_height_female, list_weight_female = [], [], [], []
    
    total = len(y)
    #get number of male and female and sum of height and weight for male and female, and list of information
    for gender in range(total):
      if y[gender] == 1:
       
        # Adding the height from the male sample to the sum of heights
        sum_height_male = sum_height_male + x[gender][0]
        
        # Appending to the list of heights
        list_height_male.append(x[gender][0])
        
        # Adding the weight from the male sample to the sum of weights
        sum_weight_male = sum_weight_male + x[gender][1]
        
        # Appending to the list of weights
        list_weight_male.append(x[gender][1])
        
        # Adding +1 to get the total number of males
        total_male_number = total_male_number + 1
        
      else:
              
        # Adding the height from the female sample to the sum of heights
        sum_height_female = sum_height_female + x[gender][0]
        
        # Appending to the list of heights
        list_height_female.append(x[gender][0])
        
        # Adding the weight from the female sample to the sum of weights
        sum_weight_female = sum_weight_female + x[gender][1]
        
        # Appending to the list of weights
        list_weight_female.append(x[gender][1])
        
        # Adding +1 to get the total number of females
        total_female_number = total_female_number + 1
    
    # total number of samples 
    num_total = total_male_number + total_female_number
        
    # mu = [sum_height/total_number, sum_weight/total_number]
    mu_male = [sum_height_male/total_male_number, sum_weight_male/total_male_number]
    mu_female = [sum_height_female/total_female_number, sum_weight_female/total_female_number]
    
    # Now calcultaing the mu value for both the genders combined
    mu_height = (sum_height_male + sum_height_female) / num_total
    mu_weight = (sum_weight_male + sum_weight_female) / num_total

    #initialization
    cov_top_left, cov_top_right, cov_bottom_right = 0, 0, 0

    # Calculating each value of the cov matrix for both male and female combined
    for gender in range(total):
      # top left value
      cov_top_left = cov_top_left + (x[gender][0]-mu_height) ** 2
      # top right value
      cov_top_right = cov_top_right + (x[gender][0]-mu_height) * (x[gender][1]-mu_weight)
      # bottom right value
      cov_bottom_right = cov_bottom_right + (x[gender][1]-mu_weight) ** 2
      
    # As we know that the covariance matrix is symmetric, the top left and bottom right values are same
    cov_bottom_left = cov_top_right

    # Now we can calculate the covariance matrix
    cov = [[cov_top_left/num_total, cov_top_right/num_total], [cov_bottom_left/num_total, cov_bottom_right/num_total]]

    #initialization
    cov_male_topleft, cov_male_topright, cov_male_bottomright, cov_female_topleft, cov_female_topright, cov_female_bottomright = 0, 0, 0, 0, 0, 0 

    #calculate cov matrix for male and female
    for gender in range(total):
      if y[gender] == 1:
        cov_male_topleft = cov_male_topleft + (x[gender][0]-mu_male[0]) ** 2
        cov_male_topright = cov_male_topright + (x[gender][0]-mu_male[0]) * (x[gender][1]-mu_male[1])
        cov_male_bottomright = cov_male_bottomright + (x[gender][1]-mu_male[1]) ** 2
      else:
        cov_female_topleft = cov_female_topleft + (x[gender][0]-mu_female[0]) ** 2
        cov_female_topright = cov_female_topright + (x[gender][0]-mu_female[0]) * (x[gender][1]-mu_female[1])
        cov_female_bottomright = cov_female_bottomright + (x[gender][1]-mu_female[1]) ** 2

    # As we know that the covariance matrix is symmetric, the top right and bottom left values are same
    cov_male_bottomleft = cov_male_topright
    cov_female_bottomleft = cov_female_topright

    cov_male = [[cov_male_topleft/total_male_number, cov_male_topright/total_male_number], [cov_male_bottomleft/total_male_number, cov_male_bottomright/total_male_number]]
    cov_female = [[cov_female_topleft/total_female_number, cov_female_topright/total_female_number], [cov_female_bottomleft/total_female_number, cov_female_bottomright/total_female_number]]

    # print(mu_male)
    # print(mu_female)
    # print(cov)
    # print(cov_male)
    # print(cov_female)

    #LDA
    #plotting datapoints
    plt.scatter(list_height_male, list_weight_male, color = 'green')
    plt.scatter(list_height_female, list_weight_female, color = 'orange')
    
    #plotting gradient
    x_value = np.linspace(50, 80, 100)   
    y_value = np.linspace(80, 280, 100)   
    X, Y = np.meshgrid(x_value, y_value)
    LDA_m, QDA_m, LDA_f, QDA_f = [], [], [], [] 
    
    
    # calculate the density for each point
    for i in range(100):
      samples = np.concatenate((X[i].reshape(100, 1), Y[i].reshape(100, 1)), 1)
     
     # LDAs
      LDA_f.append(util.density_Gaussian(mu_female,cov,samples))
      LDA_m.append(util.density_Gaussian(mu_male,cov,samples))
      # QDAs
      QDA_f.append(util.density_Gaussian(mu_female,cov_female,samples))
      QDA_m.append(util.density_Gaussian(mu_male,cov_male,samples))
      
    
    
    
    
    
    # Graph for LDA
    plt.contour(X,Y,LDA_m,colors='b')
    plt.contour(X,Y,LDA_f,colors='r')

    LDA_boundary = np.asarray(LDA_m) - np.asarray(LDA_f)
    plt.contour(X,Y,LDA_boundary,0)
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.legend(['Male', 'Female'],loc = 'upper left') 
    plt.title('LDAs Height vs. Weight (Training Data)')
    plt.savefig("lda.pdf")
    plt.show()
   

    # graph for QDA
    plt.scatter(list_height_male, list_weight_male, color = 'green')
    plt.scatter(list_height_female, list_weight_female, color = 'orange')
    
    plt.contour(X,Y,QDA_m,colors='b')
    plt.contour(X,Y,QDA_f,colors='r')
    
    QDA_boundary = np.asarray(QDA_m) - np.asarray(QDA_f)
    plt.contour(X,Y,QDA_boundary,0)
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('QDAs Height vs. Weight (Training Data)')
    plt.legend(['Male', 'Female'],loc = 'upper left') 
    plt.savefig("qda.pdf")
    plt.show()
    
    return (np.asarray(mu_male),np.asarray(mu_female),np.asarray(cov),np.asarray(cov_male),np.asarray(cov_female))
    

def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples   
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    ### TODO: Write your code here
    
    #Calculation LDAs
    LDA_m = np.dot(mu_male.T, np.dot(np.linalg.inv(cov), x.T)) - 0.5*np.dot(mu_male.T, np.dot(np.linalg.inv(cov), mu_male))
    LDA_f = np.dot(mu_female.T, np.dot(np.linalg.inv(cov), x.T)) - 0.5*np.dot(mu_female.T, np.dot(np.linalg.inv(cov), mu_female))
    
    right_LDA = 0
    total = len(y)
    
    for i in range(total):
      if (LDA_m[i]>=LDA_f[i] and y[i]==1) or (LDA_m[i]<=LDA_f[i] and y[i]==2):
        right_LDA = right_LDA + 1
        
    # Calculate the misclassification rate for LDAs
    mis_r_LDA = 1-right_LDA/total
    
    #QDA calculation
    QDA_m, QDA_f = [], []
    
    for i in range(x.shape[0]):
      QDA_m.append(- 0.5*np.log(np.linalg.det(cov_male)) - 0.5*np.dot(x[i], np.dot(np.linalg.inv(cov_male), x[i].T)) + np.dot(mu_male.T, np.dot(np.linalg.inv(cov_male), x[i].T)) - 0.5*np.dot(mu_male.T, np.dot(np.linalg.inv(cov_male), mu_male)))
      QDA_f.append( - 0.5*np.log(np.linalg.det(cov_female)) - 0.5*np.dot(x[i], np.dot(np.linalg.inv(cov_female), x[i].T)) + np.dot(mu_female.T, np.dot(np.linalg.inv(cov_female), x[i].T)) - 0.5*np.dot(mu_female.T, np.dot(np.linalg.inv(cov_female), mu_female)))   
    
    # right_QDA records the number of correct predictions
    right_QDA = 0   
    
    for i in range(total):
      if (QDA_m[i]>=QDA_f[i] and y[i]==1) or (QDA_m[i]<=QDA_f[i] and y[i]==2):
        right_QDA = right_QDA + 1
    mis_r_QDA = 1-right_QDA/total

    
    return (mis_r_LDA, mis_r_QDA)


if __name__ == '__main__':
    
       
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('C:/Users/mehul/Downloads/ldaqda/trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('C:/Users/mehul/Downloads/ldaqda/testHeightWeight.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)
    
    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)