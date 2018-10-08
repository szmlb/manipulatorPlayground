import matplotlib.pyplot as plt
import numpy as np
import math
from collections import namedtuple

class LinkParams:

    def __init__(self, dof):
        self.l = 0.0
        self.c = 0.0
        self.m = 0.0
        self.px = 0.0
        self.py = 0.0
        self.pz = 0.0
        self.Lxx = 0.0
        self.Lxy = 0.0
        self.Lxz = 0.0
        self.Lyy = 0.0
        self.Lyz = 0.0
        self.Lzz = 0.0
        self.jp = np.zeros(dof)

class TwoLinkArm:

    def __init__(self, genModelFlag):

        self.step_time = 0.01
        self.dof = 2

        # plot
        plt.ion()
        self.fig = plt.figure()

        # variables
        self.q = np.zeros(self.dof)
        self.dq = np.zeros(self.dof)
        self.ddq = np.zeros(self.dof)
        self.tau = np.zeros(self.dof)
        self.q_cmd = np.zeros(self.dof)
        self.dq_cmd = np.zeros(self.dof)
        self.ddq_cmd = np.zeros(self.dof)

        self.tcp_pos = np.zeros(self.dof) # tool center point
        self.tcp_pos_cmd = np.zeros(self.dof) # tool center point
        self.tcp_vel_cmd = np.zeros(self.dof) # tool center point

        self.jaco = np.zeros([self.dof, self.dof])

        self.link_params = []

        # link1
        self.link_params.append(LinkParams(self.dof))
        print(self.link_params[0].l)
        self.link_params[0].l = 0.1
        self.link_params[0].c = self.link_params[0].l / 2.0
        self.link_params[0].m = 0.1
        self.link_params[0].px = self.link_params[0].m * self.link_params[0].c
        self.link_params[0].py = 0.0
        self.link_params[0].pz = 0.0
        self.link_params[0].Lxx = 0.0
        self.link_params[0].Lxy = 0.0
        self.link_params[0].Lxz = 0.0
        self.link_params[0].Lyy = self.link_params[0].m * self.link_params[0].l**2 / 12.0 + self.link_params[0].m * self.link_params[0].c**2
        self.link_params[0].Lyz = 0.0
        self.link_params[0].Lzz = self.link_params[0].m * self.link_params[0].l**2 / 12.0 + self.link_params[0].m * self.link_params[0].c**2
        self.link_params[0].jp = np.zeros(self.dof)

        # link2
        self.link_params.append(LinkParams(self.dof))
        self.link_params[1].l = 0.1
        self.link_params[1].c = self.link_params[1].l / 2.0
        self.link_params[1].m = 0.1
        self.link_params[1].px = self.link_params[1].m * self.link_params[1].c
        self.link_params[1].py = 0.0
        self.link_params[1].pz = 0.0
        self.link_params[1].Lxx = 0.0
        self.link_params[1].Lxy = 0.0
        self.link_params[1].Lxz = 0.0
        self.link_params[1].Lyy = self.link_params[1].m * self.link_params[1].l**2 / 12.0 + self.link_params[1].m * self.link_params[1].c**2
        self.link_params[1].Lyz = 0.0
        self.link_params[1].Lzz = self.link_params[1].m * self.link_params[1].l**2 / 12.0 + self.link_params[1].m * self.link_params[1].c**2
        self.link_params[1].jp = np.zeros(self.dof)

        # equation of motion
        self.dyn_params = []
        for i in range(self.dof):
            self.dyn_params.append(self.link_params[i].Lxx)
            self.dyn_params.append(self.link_params[i].Lxy)
            self.dyn_params.append(self.link_params[i].Lxz)
            self.dyn_params.append(self.link_params[i].Lyy)
            self.dyn_params.append(self.link_params[i].Lyz)
            self.dyn_params.append(self.link_params[i].Lzz)
            self.dyn_params.append(self.link_params[i].px)
            self.dyn_params.append(self.link_params[i].py)
            self.dyn_params.append(self.link_params[i].pz)
            self.dyn_params.append(self.link_params[i].m)

        self.Meq = np.zeros([self.dof, self.dof])
        self.ceq = np.zeros([self.dof])
        self.geq = np.zeros([self.dof])
        self.M_matrix(self.dyn_params, self.q)
        self.c_vector(self.dyn_params, self.q, self.dq)
        self.g_vector(self.dyn_params, self.q)

    def M_matrix(self, dyn_params, q):

        M_out = [0]*4

        x0 = 0.1*math.cos(q[1])
        x1 = 0.1*math.sin(q[1])
        x2 = -dyn_params[17]
        x3 = dyn_params[15] + dyn_params[16]*x0 + x1*x2
        M_out[0] = dyn_params[5] + x0*(dyn_params[16] + dyn_params[19]*x0) + x1*(dyn_params[19]*x1 + x2) + x3
        M_out[1] = x3
        M_out[2] = x3
        M_out[3] = dyn_params[15]

        M_out = np.array(M_out)
        self.Meq = M_out.reshape(2, 2)

    def c_vector(self, dyn_params, q, dq):

        c_out = [0]*2

        x0 = -dq[0]**2
        x1 = math.sin(q[1])
        x2 = -0.1*x0*x1
        x3 = math.cos(q[1])
        x4 = 0.1*x0*x3
        x5 = dyn_params[16]*x2 - dyn_params[17]*x4
        x6 = -(dq[0] + dq[1])**2

        c_out[0] = 0.1*x1*(dyn_params[16]*x6 + dyn_params[19]*x4) + 0.1*x3*(dyn_params[17]*x6 + dyn_params[19]*x2) + x5
        c_out[1] = x5

        self.ceq = np.array(c_out)

    def g_vector(self, dyn_params, q):

        g_out = [0]*2

        x0 = 9.81*math.sin(q[0])
        x1 = -x0
        x2 = math.sin(q[1])
        x3 = 9.81*math.cos(q[0])
        x4 = math.cos(q[1])
        x5 = x1*x2 + x3*x4
        x6 = x0*x4 + x2*x3
        x7 = dyn_params[16]*x5 - dyn_params[17]*x6

        g_out[0] = 0.1*dyn_params[19]*x2*x6 + 0.1*dyn_params[19]*x4*x5 + dyn_params[6]*x3 + dyn_params[7]*x1 + x7
        g_out[1] = x7

        self.geq = np.array(g_out)

    def control(self, time):

        if time < 1.0:
            self.q_cmd[0] = 0.0
            self.q_cmd[1] = np.pi / 2.0

            self.tcp_pos_cmd[0] = 0.1
            self.tcp_pos_cmd[1] = 0.1

        else:
            self.tcp_pos_cmd[0] = 0.1 + 0.05 * np.sin(2.0 * np.pi * 0.5 * time)
            self.tcp_pos_cmd[1] = 0.1 + 0.05 * np.cos(2.0 * np.pi * 0.5 * time)

            self.tcp_vel_cmd[0] = 15.0 * (self.tcp_pos_cmd[0] - self.tcp_pos[0])
            self.tcp_vel_cmd[1] = 15.0 * (self.tcp_pos_cmd[1] - self.tcp_pos[1])

            self.dq_cmd = np.linalg.inv(self.jaco).dot(self.tcp_vel_cmd)
            self.q_cmd = self.q_cmd + self.dq_cmd * self.step_time

            self.ddq_cmd[0] = 10.0 * (self.q_cmd[0] - self.q[0]) + 1.0 * (self.dq_cmd[0] - self.dq[0])
            self.ddq_cmd[1] = 10.0 * (self.q_cmd[1] - self.q[1]) + 1.0 * (self.dq_cmd[1] - self.dq[1])

        self.tau = self.Meq.dot(self.ddq_cmd) + self.ceq + self.geq

    def update(self):

        # dynamics
        self.M_matrix(self.dyn_params, self.q)
        self.c_vector(self.dyn_params, self.q, self.dq)
        self.g_vector(self.dyn_params, self.q)

        self.ddq = np.linalg.inv(self.Meq).dot( self.tau - self.ceq - self.geq  )
        self.dq = self.dq + self.ddq * self.step_time
        self.q = self.q + self.dq * self.step_time

        # kinematics
        self.link_params[0].jp[0] = 0.0
        self.link_params[0].jp[1] = 0.0

        self.link_params[1].jp[0] = self.link_params[0].jp[0] + self.link_params[0].l * np.cos(self.q[0])
        self.link_params[1].jp[1] = self.link_params[0].jp[1] + self.link_params[0].l * np.sin(self.q[0])

        self.tcp_pos[0] = self.link_params[0].jp[0] + self.link_params[1].jp[0] + self.link_params[1].l * np.cos(self.q[0] + self.q[1])
        self.tcp_pos[1] = self.link_params[0].jp[1] + self.link_params[1].jp[1] + self.link_params[1].l * np.sin(self.q[0] + self.q[1])

        self.jaco[0, 0] = -self.link_params[0].l * np.sin(self.q[0]) - self.link_params[1].l * np.sin(self.q[0] + self.q[1])
        self.jaco[0, 1] = -self.link_params[1].l * np.sin(self.q[0] + self.q[1])
        self.jaco[1, 0] = self.link_params[0].l * np.cos(self.q[0]) + self.link_params[1].l * np.cos(self.q[0] + self.q[1])
        self.jaco[1, 1] = self.link_params[1].l * np.cos(self.q[0] + self.q[1])

    def simulation(self, sim_time):

        for i in range(int(sim_time/self.step_time)):

            time = i * self.step_time

            self.control(time)
            self.update()
            self.plot()

    def plot(self):

        # プロット
        plt.cla() # 現在のプロットを削除

        plt.plot([self.link_params[0].jp[0], self.link_params[1].jp[0]], [self.link_params[0].jp[1], self.link_params[1].jp[1]], 'k-') # リンク1の直線
        plt.plot([self.link_params[1].jp[0], self.tcp_pos[0]], [self.link_params[1].jp[1], self.tcp_pos[1]], 'k-') # リンク2の直線

        plt.plot(self.link_params[0].jp[0], self.link_params[0].jp[1], 'ro') # 関節1の位置
        plt.plot(self.link_params[1].jp[0], self.link_params[1].jp[1], 'ro') # 関節2の位置
        plt.plot(self.tcp_pos[0], self.tcp_pos[1], 'ro') # 手先の位置
        plt.plot(self.tcp_pos_cmd[0], self.tcp_pos_cmd[1], 'g*') # 手先の位置

        plt.xlim(-0.1, 0.3) # xlimit
        plt.ylim(-0.2, 0.2) # ylimit

        plt.show()
        plt.pause(self.step_time)

def main():

    rob = TwoLinkArm(genModelFlag=False)
    rob.simulation(5.0)

if __name__ == "__main__":
    main()
