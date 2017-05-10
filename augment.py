from math import cos,tan,sqrt,sin,asin


#==========================================================
# Calculate steering angle based on shifted/translate image
#==========================================================

# From ALVINN: time used to calculate lookahead distance l
l_time = 2.5

#From udacity buick (in feet)
wheelbase = 9.33

# r_p = previous turning radius
# d_p = A to T distance
# d = B to T distance
# s = shift in feet
# speed = speed in FEET per SECOND
# l = lookahead distance

def calc_r_p(steering_angle):
    return(wheelbase/sin(steering_angle))

def calc_r(steering_angle,speed,s,t):
    r_p = calc_r_p(steering_angle)
    l = speed*l_time
    d_p = r_p - sqrt(r_p**2-l**2)
    d = cos(t)*(d_p + s + l*tan(t))
    return (l**2 + d**2)/(2*d)

def new_steering_angle(steering_angle,speed,s,t):
    speed *= 1.4666666
    r = calc_r(steering_angle,speed,s,t)
    return asin(wheelbase/r)


#==========================
# Shift and translate image
#==========================





#=======================================
# Reflect image and negate steering angle
#======================================




