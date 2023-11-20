import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

import cv2
from InvertedPendulum import InvertedPendulum


g = 9.81
L = 10

Time_STEP = 0.05

initial_values = [np.pi / 3, 0]  

g = 9.8  
L = 1.0  


t_current = 0
dt = 0.05 

#PID
kp =1
kd =0
ki =0


def simple_pendulum(t, y):
    return [y[1], -(g / L) * np.sin(y[0])]

class PID():
    def __init__(self , kp , ki , kd , target):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setPoint = target
        self.error = 0
        self.integralError = 0
        self.error_last = 0
        self.derrivative_error = 0
        self.output = 0
        
    def compute(self , pos):
        self.error = self.setPoint - pos
        self.integralError += self.error*Time_STEP
        self.derrivative_error += (self.error - self.error_last)/Time_STEP
        self.error_last = self.error
        self.output = self.kp * self.error + self.ki*self.integralError + self.kd*self.derrivative_error
        return self.output


def cycle(t_stop=10 , kp=1 , kd=1 , ki=1 , initial_values = [np.pi / 4, 0]  ):
    
    t_current = 0
    Controller = PID(kp,kd,ki, np.deg2rad(180))
    initial_conditions = initial_values
    time_points = [t_current]
    solution_values = [initial_conditions]
    theta_val = []
    theta_val.append(np.rad2deg(solution_values[0][0]))
    theta_dot_val = []
    theta_dot_val.append(solution_values[0][1])
    
    while t_current < t_stop:
        
        t_span = (t_current, t_current + dt)
    
        sols = solve_ivp(fun=lambda t, y: simple_pendulum(t, y), t_span=t_span, y0=initial_conditions)
        
        sol = [sols.y[0, -1], sols.y[1, -1]] 
        t_current += dt
        
        theta = sol[0]
        
        #sending to controller
        theta = Controller.compute(sol[0])
        
        initial_conditions = [theta, sol[1]]  
        time_points.append(t_current)
        solution_values.append(initial_conditions)
        theta_val.append(np.rad2deg(theta))
        theta_dot_val.append(np.rad2deg(sol[1]))
        
    return (theta_val , theta_dot_val)


syst = InvertedPendulum()

def simulation(time_net,kp , kd , ki , show_diagram=False):
    res = cycle(time_net,kp=kp , kd=kd,ki = ki)
    
    fig = plt.figure(figsize=(30, 15))
    plt.plot(res[0] , 'r' , label= 'theta' , linewidth=2.5)
    plt.yticks(fontsize=14)
    plt.suptitle(str([kp , kd , ki , res[0][-1] , res[1][-1]]), fontsize=18)
    plt.plot(res[1] , 'b' , label='velocity' , linewidth=2.5)
    plt.legend()
    plt.grid()
    plt.show()
    
    if(show_diagram):
        for i in range(int(time_net/Time_STEP)):
        
            rendered = syst.step( [0,1, np.deg2rad( res[0][i]-90), np.deg2rad( res[1][i])])
            cv2.imshow( 'im', rendered )

            if cv2.waitKey(30) == ord('q'):
                break
            
    cv2.destroyAllWindows()
    
    
simulation(150 , 0.5,0.4,0.003 , False)


