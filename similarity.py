import numpy as np

class string_similarity:
  """
  A class that 
  """

  def __init__(self,):
    self.scaler_functions = {'levenshtein_metric':self.scalar_leve_distance,'damerau_levenshtein_metric':self.scaler_damerau_levenshtein}


  def scalar_leve_distance(self,word1:str,word2:str) -> int:
    """
    Calculate distance between two adjacent words
    """
    # Get matrix shape
    rows    = len(word1) + 1 
    columns = len(word2) + 1 

    # Initialize distance matrix
    dist_matrix = np.zeros((rows,columns))

    for i in range(rows) :
      dist_matrix[i,0] = i
    
    for j in range(columns) :
      dist_matrix[0,j] = j
    
    # Calculate number of changes 
    for i in range(1, rows):  
      for j in range(1,columns):

        if word1[i-1] == word2[j-1]:
          dist_matrix[i,j] = min(dist_matrix[i-1,j-1], dist_matrix[i-1,j]+1,dist_matrix[i,j-1]+1)
        
        else:
          dist_matrix[i,j] = min(dist_matrix[i-1,j]+1, dist_matrix[i-1,j-1]+1,dist_matrix[i,j-1]+1)

    return int(dist_matrix[-1,-1])


  def scaler_damerau_levenshtein(s1:str, s2:str) -> int:
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition

    return d[lenstr1-1,lenstr2-1]

  def scaler_metrics(self,func,s1:list,s2:list) -> float:
    assert isinstance(s1,list) and isinstance(s2,list), "Input should be of type" 
    assert len(s1) == len(s2), "Lengths of the two iterators are not equal"

    floatn = None
    floats = []
    
    for str1,str2 in zip(s1,s2):
      floatn = 1.0 - (func(str1,str2) / max(len(str1),len(str2)))
      floats.append(floatn)
    
    return float(sum(floats)/len(floats))
    
  def levenshtein_metric(self,s1:list,s2:list) -> float:
    result = self.scaler_metrics(self.scaler_functions['levenshtein_metric'],s1,s2)
    return result 

  def damerau_levenshtein_metric(self,s1:list,s2:list) -> float:
    result = self.scaler_metrics(self.scaler_functions['damerau_levenshtein_metric'],s1,s2)
    return result 

