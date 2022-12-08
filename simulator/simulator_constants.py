#system parameters
## manipulator parameters
m = 1.17  #mass - nominal 1 (between 0.8 and 1.2)
m_n = 1
m_min = 0.8
m_max = 1.2

l = 1.57  #length - nominal 1.5 (between 1.25 and 1.75)
l_n = 1.5
l_min = 1.25
l_max = 1.75

d = 0.02  # diameter

J = m*l**2/3 #mass moment of inertia nominal 1x1.5x1.5/3 (between 0.8 x 1.25 x 1.25/3 and 1.2 x 1.75 x 1.75/3 )

J_n = m_n*l_n**2/3
J_min = m_min*l_min**2/3
J_max = m_max*l_max**2/3

g = 9.82 # acceleration due to gravity #between 9.78 to 9.83
g_n = 9.805
g_min = 9.78
g_max = 9.83

c = 0.97 # damping constant
c_n = 1
c_min = 0.8
c_max = 1.2

mgl = m*g*l
mgl_n = m_n*g_n*l_n
mgl_min = m_min*g_min*l_min
mgl_max = m_max*g_max*l_max

# dry friction parameters
mu_s = 0.6 #coefficeint of static friction
mu_d = 0.5 #coefficient of dynamic friction
vs = 0.01 #some parameter
gamma = 1 #some parameter


# sample time
dt = 0.02 #1e-5

# drag parameters
p = 1

m_test = 1.2
l_test = 1.53
g_test = 9.8
c_test = 1
J_test = m_test*l_test**2/3
mgl_test = m_test*g_test*l_test
