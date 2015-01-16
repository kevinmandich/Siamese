

Delta1, Delta2, Delta3 = 0, 0, 0
for i, row in enumerate(X):
    aIn, zH1, aH1, zH2, aH2, zOut, aOut = self._forward(row, t1, t2, t3)
    
    ## back-propagation
    dOut = aOut - Y[i, :].T
    dH2 = np.dot(t3f.T, dOut) * self.sigmoid_deriv(zH2) # zH[1]
    dH1 = np.dot(t2f.T, dH2)  * self.sigmoid_deriv(zH1) # zH[0]

    Delta3 += np.dot(dOut[np.newaxis].T, aH2[np.newaxis]) # aH[1]
    Delta2 += np.dot(dH2[np.newaxis].T,  aH1[np.newaxis]) # aH[0]
    Delta1 += np.dot(dH1[np.newaxis].T,  aIn[np.newaxis])
    
Theta1_grad = (1 / m) * Delta1
Theta2_grad = (1 / m) * Delta2
Theta3_grad = (1 / m) * Delta3

## apply regularization
if lambda_ != 0:
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (lambda_ / m) * t1f
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (lambda_ / m) * t2f
    Theta3_grad[:, 1:] = Theta3_grad[:, 1:] + (lambda_ / m) * t3f




Delta1, Delta2 = 0, 0
for i, row in enumerate(X):
    a1, z2, a2, z3, a3 = self._forward(row, t1, t2)
    
    ## back-propagation
    d3 = a3 - Y[i, :].T
    d2 = np.dot(t2f.T, d3) * self.sigmoid_deriv(z2)  
         
    Delta2 += np.dot(d3[np.newaxis].T, a2[np.newaxis])
    Delta1 += np.dot(d2[np.newaxis].T, a1[np.newaxis])
    
Theta1_grad = (1 / m) * Delta1
Theta2_grad = (1 / m) * Delta2

## apply regularization
if lambda_ != 0:
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (lambda_ / m) * t1f
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (lambda_ / m) * t2f

return self.pack_thetas(Theta1_grad, Theta2_grad)
