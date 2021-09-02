import matplotlib.pyplot as plt
import numpy as np

# plt.style.use('seaborn-whitegrid')


# p1=np.array([0,1,2])
# p2=np.array([10,9,8])
# l1 = np.linspace(0,1,11)

# segment = p1+(p2-p1)*l1[:,None]
#
#
# # print(segment[1,:])
# #
# #
# # input("press")
#
#
# plt.scatter(segment[:,0],segment[:,1])
# plt.show()
# input("press")


pos_init = np.array([0, 0, 0])
pos_goal = np.array([2, 1, 0])
pos_ob = np.array([1, 2, 0])

# go back to the line
unit_vector_1 = (pos_ob - pos_init) / np.linalg.norm(pos_ob - pos_init)
unit_vector_2 = (pos_goal - pos_init) / np.linalg.norm(pos_goal - pos_init)
cosalpha = np.dot(unit_vector_1, unit_vector_2)

n = unit_vector_2 * cosalpha * np.linalg.norm(pos_ob - pos_init)
vec_d = (pos_ob - pos_init) - n
pos_goal2 = pos_ob - vec_d

# distance object to the segment
d = np.linalg.norm(np.cross(pos_init[0:2] - pos_goal[0:2], pos_goal[0:2] - pos_goal2[0:2])) / np.linalg.norm(pos_init[0:2] - pos_goal[0:2])

print("d", d)
plt.plot([pos_init[0], pos_goal[0]], [pos_init[1], pos_goal[1]])
plt.plot([pos_init[0], pos_ob[0]], [pos_init[1], pos_ob[1]])
plt.plot([pos_ob[0], pos_goal2[0]], [pos_ob[1], pos_goal2[1]])
plt.plot([pos_init[0], n[0]], [pos_init[1], n[1]])
plt.show()
input("press")
